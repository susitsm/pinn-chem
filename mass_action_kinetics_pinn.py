#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation=nn.Tanh):
        super().__init__()
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

class Complex:
    def __init__(self, cmplx: str):
        self.string_form = cmplx
        coefficients = {}
        if cmplx != "0":
            for species in cmplx.split("+"):
                if "*" in species:
                    coeff, species = species.split("*")
                    coefficients[species] = int(coeff)
                else:
                    coefficients[species] = 1
        self.coefficients = coefficients
        
    def __hash__(self):
        return self.string_form.__hash__()
    
    def __eq__(self, other):
        return self.coefficients == other.coefficients
    
    def __repr__(self):
        return self.string_form
        
    def is_zero(self) -> bool:
        return len(self.coefficients) == 0
    
    def species(self):
        return self.coefficients.keys()
    
    def __getitem__(self, species):
        return self.coefficients.get(species, 0)

class ReactionStep:
    def __init__(self, reactant: Complex, product: Complex, rate: float = None):
        self.reactant = reactant
        self.product = product
        self.rate = rate
    
    def __hash__(self):
        return (self.reactant, self.product).__hash__()
    
    def __eq(self, other):
        return self.reactant == other.reactant and self.product == other.product
    
    def __repr__(self):
        return f"{self.reactant} -> {self.product}"

class ReactionRates:
    def __init__(self, rates: list, dtype = torch.float32):
        """
        each element of rates should be a float, 3 length tuple or list like
        
        if the nth element is a float or int the reaction rate of the nth reaction is fixed
        if the nth element is a 3 length tuple, the nth reaction's reaction rates are a torch.linspace with the tuple's element as arguments
        otherwise the nth element is interpreted as a list of reaction rates
        """
        free_variables = []
        canonized = []
        for (i, rate) in enumerate(rates):
            if type(rate) in (float, int):
                canonized.append(torch.tensor([float(rate)], dtype = dtype))
            elif type(rate) is tuple and len(rate) == 3:
                canonized.append(torch.linspace(*rate, dtype = dtype))
                free_variables.append(i)
            else:
                canonized.append(torch.tensor(rate, dtype = dtype))
                free_variables.append(i)
        # the k-s with multiple values
        self.free_variables = free_variables
        self.canonized = canonized
        self.meshgrid = torch.stack(torch.meshgrid(canonized), axis = len(canonized))
    
    def input_meshgrid(self, t):
        axes = [t]
        for i in self.free_variables:
            axes.append(self.canonized[i])
        return torch.stack(torch.meshgrid(axes), axis = self.n_parameters()+1)
    
    def n_parameters(self):
        return len(self.free_variables)

class ReactionNetwork:
    def __init__(self, reactions: str, reaction_rates = None):
        """
        reactions is a comma separated list of reactions. Whitespaces are ignored.
        Example reactions:
            "A -> 0, A <-> B"
            "0 <- A <-> 2*B -> C"
        
        TODO handle inflow in calculations, 0 -> A
        """
        reactions0 = "".join(reactions.split())
        reactions1 = reactions0.split(",")
        reactions2 = [reaction.split("-") for reaction in reactions1]
        def canonical_reactions(c1: str, c2: str) -> list[(Complex, Complex)]:
            left, right = False, False
            if c1[0] == ">":
                c1 = c1[1:]
            if c1[-1] == "<":
                c1 = c1[:-1]
                left = True
            cmplx1 = Complex(c1)
            if c2[0] == ">":
                right = True
                c2 = c2[1:]
            if c2[-1] == "<":
                c2 = c2[:-1]
            cmplx2 = Complex(c2)
            res = []
            if right:
                res.append(ReactionStep(cmplx1, cmplx2))
            if left:
                res.append(ReactionStep(cmplx2, cmplx1))
            return res
        
        self.reactions = [reaction for complexes in reactions2 for leftright in map(lambda x: canonical_reactions(*x), zip(complexes[:-1], complexes[1:])) for reaction in leftright]
        self.complexes = set(cmplx for reaction in self.reactions for cmplx in (reaction.reactant, reaction.product))
        self.species = list(sorted(set(species for cmplx in self.complexes for species in cmplx.species())))
        self.species_index = {species:i for (i, species) in enumerate(self.species)}
        assert len(reaction_rates) == len(self.reactions), f"number of reaction rates and reactions do not mach {len(reaction_rates)} {len(self.reactions)}"
        self.reaction_rates = ReactionRates(reaction_rates) if reaction_rates is not None else None
        
        # calculate the stoichiometric matrix
        self.alpha = torch.zeros((len(self.species),len(self.reactions)))
        self.beta = torch.zeros((len(self.species),len(self.reactions)))
        for (i, reaction) in enumerate(self.reactions):
            for species, coeff in reaction.reactant.coefficients.items():
                self.alpha[self.species_index[species],i] = coeff
            for species, coeff in reaction.product.coefficients.items():
                self.beta[self.species_index[species],i] = coeff

        self.gamma = self.beta-self.alpha

    def n_param_reaction_rate(self):
        """
        Number of k-s with more than 1 possible value
        """
        return self.reaction_rates.n_parameters()
    
    def n_species(self) -> int:
        return len(self.species)
    
    def n_reactions(self) -> int:
        return len(self.reactions)
    
    def mass_action_dxdt(self, x):
        """
        calculate dx/dt at x by evaluating the kinetic differential equation. X must have shape (n_data, n_species)
        
        """
        return mass_action_dxdt_torch(x, self.alpha, self.beta, self.reaction_rates.meshgrid)

    def mass_action_kinetics_loss(self, model, input_physics, t_physics):
        """
        Calculate the mass action kinetics loss by comparing dx/dt and the kinetic de rhs for model
        """
        xph = model(input_physics)
        xph_3d = xph.view(xph.shape[0],-1,xph.shape[-1])
        xph_2d = xph.view(xph.shape[0],-1)
        # TODO do this with a vectorized operation
        dxdt = torch.cat([torch.autograd.grad(xph_2d[:,i], t_physics, torch.ones_like(xph_2d[:,i]), retain_graph=True)[0] for i in range(xph_2d.shape[1])], axis=1).view(xph_3d.shape)
        dxdt_mass_action = self.mass_action_dxdt(xph_3d)
        assert dxdt.shape == xph_3d.shape, f"dxdt: {dxdt.shape}, dxdt_ma: {dxdt_mass_action.shape}"
        assert dxdt.shape == dxdt_mass_action.shape, f"dxdt: {dxdt.shape}, dxdt_ma: {dxdt_mass_action.shape}"
        
        # saved for plotting
        self.dxdt = dxdt
        self.dxdt_mass_action = dxdt_mass_action
        
        return mse_loss(dxdt, dxdt_mass_action)
    
    def __repr__(self):
        return f"ReactionNetwork({self.reactions})"
    
    def numeric_solve_mass_action_kinetics_de_(self, initial_concentration, time, t_eval):
        """
        Solve the kinetic de of self starting from initial_concentration on time interval (0,time)
        return the solution at points t_eval for every reaction rate combination
        """
        meshgrid = self.reaction_rates.meshgrid
        res = np.empty((t_eval.shape[0],) + meshgrid.shape[:-1] + (initial_concentration.shape[0],))
        # solve for every reaction rate vector
        for ind, reaction_rates in deep_iter(meshgrid, max_depth=meshgrid.dim()-1):
            rhs = MassActionKineticsRightHandSide(self.alpha.numpy(), self.beta.numpy(), reaction_rates.numpy())
            sol = solve_ivp(rhs, (0,time), initial_concentration, vectorized = True, t_eval=t_eval).y.T
            res[(slice(None),)+ind+(slice(None),)] = sol
        return (res, sol)
    
    def numeric_solve_mass_action_kinetics_de(self, initial_concentration, time, t_eval):
        (res, sol) = self.numeric_solve_mass_action_kinetics_de_(initial_concentration, time, t_eval)
        return res

class MassActionKineticsRightHandSide:
    def __init__(self, alpha, beta, reaction_rates):
        self.alpha = alpha
        self.beta = beta
        self.reaction_rates = reaction_rates
    
    def __call__(self, _t, x):
        a = mass_action_dxdt_np(x, self.alpha, self.beta, self.reaction_rates)
        return a
        
        
def mass_action_dxdt_torch(x, alpha, beta, reaction_rates):
    """
    Calculate the dx/dt using the kinetic differential equation
    
    shapes:
        x: (n_t,n_nk1*...*n_kl, n_species)
        alpha, beta: (n_species, n_reactions)
        reaction_rates: (n_k1,...,nkl,n_reactions)
    
    TODO: handle inflow, 0 -> A
    
    return shape: shape of x
    """
    n_t, n_k_prod, n_species = x.shape
    n_reactions = reaction_rates.shape[-1]
    power = torch.pow(x.view(n_t,n_k_prod,1,n_species), alpha.T.view(1,1,n_reactions,n_species))
    mass_action_reaction_rate = torch.prod(power,axis=3)*reaction_rates.view(1,n_k_prod,n_reactions)
    production_rate = torch.matmul(mass_action_reaction_rate, beta.T)
    consumption_rate = torch.matmul(mass_action_reaction_rate, alpha.T)
    
    """
    if (i+1) % 500 == 0:
        print("x",x)
        print("alpha",alpha)
        print("power",power)
        print("beta",beta)
        print("production", production_rate)
        print("consumption", consumption_rate)
    """
    return production_rate-consumption_rate

def mass_action_dxdt_np(x, alpha, beta, reaction_rates):
    """
    numpy version of mass_action_dxdt_torch
    """
    n_species, n_data = x.shape
    n_reactions = reaction_rates.shape[0]
    mass_action_reaction_rate = np.prod(np.power(x.reshape((n_species,1,n_data)),alpha.reshape((n_species,n_reactions,1))),axis=0)*reaction_rates.reshape(n_reactions,1)
    production_rate = beta.dot(mass_action_reaction_rate)
    consumption_rate = alpha.dot(mass_action_reaction_rate)
    return production_rate-consumption_rate

#alternative for nditer, but for both tensor and ndarray
def deep_iter(data, ix=tuple(), max_depth=None):
    try:
        if max_depth is not None and max_depth <= len(ix):
            for i, element in enumerate(data):
                yield ix, data
        else:
            for i, element in enumerate(data):
                yield from deep_iter(element, ix + (i,), max_depth=max_depth)
    except:
        yield ix, data

def plot_result_2d_params(training_step, params_full,x_full, params_data, x_data, x_pred, params_physics):
    """
    Plot the result when there is a single species with a single input k
    """
    pass

class PlotDataConfig:
    def __init__(self, t_plot=None, plot_interval=500, t_physics=None, t_data=None, default_plot_points=500):
        if t_plot is None:
            assert t_physics is not None and t_data is not None, "Need t_phisics and t_data to calculate default value for t_plot"
            self.t_plot = torch.linspace(min(float(t_data.min()), t_physics.min().item()), max(t_data.max().item(), t_physics.max().item()), default_plot_points).view(-1,1)
        else:
            self.t_plot = t_plot.view(-1,1)
        self.plot_interval = plot_interval

class PlotData:
    def __init__(self,
                 rn: ReactionNetwork, epoch, i,
                 total_losses: list[float], data_losses: list[float], physics_losses: list[float],
                 t_plot, x_plot_model, x_plot,
                 t_training_data, x_training_data_model, x_training_data,
                 t_training_physics, x_training_physics_model, x_training_phyisics):
        self.epoch = epoch
        self.i = i
        
        self.total_loss = total_losses[-1]
        self.data_loss = data_losses[-1]
        self.physics_loss = physics_losses[-1]
        self.total_losses = total_losses
        self.data_losses = data_losses
        self.physics_losses = physics_losses

        self.t_plot = t_plot.view(-1)
        self.x_plot_model = x_plot_model.view((-1,)+rn.reaction_rates.meshgrid.shape[:-1]+(rn.n_species(),))
        self.x_plot = x_plot.view((-1,)+rn.reaction_rates.meshgrid.shape[:-1]+(rn.n_species(),))

        self.t_training_data = t_training_data.view(-1)
        self.x_training_data_model = x_training_data_model.view((-1,)+rn.reaction_rates.meshgrid.shape[:-1]+(rn.n_species(),))
        self.x_training_data = x_training_data.view((-1,)+rn.reaction_rates.meshgrid.shape[:-1]+(rn.n_species(),))
        
        self.t_training_physics = t_training_physics.view(-1)
        self.x_training_physics_model = x_training_physics_model.view((-1,)+rn.reaction_rates.meshgrid.shape[:-1]+(rn.n_species(),))
        self.x_training_physics = x_training_phyisics.view((-1,)+rn.reaction_rates.meshgrid.shape[:-1]+(rn.n_species(),))
        
        self.reaction_network = rn

    def print_loss(self):
        print(f"Epoch {self.epoch}")
        print(f"Losses: total {self.total_loss}, data {self.data_loss}, physics {self.physics_loss}")
    
    def plot_single_species_fixed_rate(self,save_file=None):
        """
        Plot the results when there is a single species with fixed reaction rates 
        """
        assert self.x_plot.shape[2] == 1, f"More than one species in self.x_plot: {self.x_plot.shape[2]}"
        assert self.x_plot.shape[1] == 1, f"The parameter k is not fixed. Parameter meshgrid shape {self.reaction_network.reaction_rates.meshgrid.shape}"
        assert self.x_plot.view(-1,1).shape[0] == self.x_plot.shape[0], f"Shape mismatch. self.x_plot.shape should be of the form (n_plot,1,...,1) but it has shape {self.x_plot.shape}"
        plt.figure(figsize=(8,4))
        plt.plot(self.t_plot,self.x_plot.view(-1,1), color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        plt.plot(self.t_plot,self.x_plot_model.view(-1,1), color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
        plt.scatter(self.t_training_data, self.x_training_data.view(-1,1), s=60, color="tab:orange", alpha=0.4, label='Training data')
        if self.t_training_physics is not None:
            plt.scatter(self.t_training_physics, self.x_training_physics.view(-1,1), s=60, color="tab:green", alpha=0.4, 
                        label='NN kinetic DE rhs')
            plt.scatter(self.t_training_physics, self.x_training_physics_model.view(-1,1), s=60, color="tab:red", alpha=0.4, 
                        label='NN dxdt')
        l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        plt.ylabel('Concentration [mass fr.]')
        plt.xlabel('Time [s]')
        if save_file is not None:
            plt.savefig(f"{save_file}_{self.i}.png")
        plt.show()
    
    def plot_result_2d_x(self,save_file=None):
        """
        plot the result when the reaction has 2 species with fixed reaction rates
        """
        x_plot = self.x_plot.view(-1,2)
        x_plot_model = self.x_plot_model.view(-1,2)
        x_data = self.x_training_data.view(-1,2)
        x_data_model = self.x_training_data_model.view(-1,2)
        plt.figure(figsize=(5,5))
        plt.plot(x_plot[:,0],x_plot[:,1], color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        plt.plot(x_plot_model[:,0],x_plot_model[:,1], color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
        plt.scatter(x_data[:,0], x_data[:,1], s=60, color="tab:orange", alpha=0.4, label='Training data')
        plt.scatter(x_data_model[:,0], x_data_model[:,1], s=60, color="yellow", alpha=0.4, label='Training data prediction')
        l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        plt.ylabel(f'Concentration [mass fr.] {self.reaction_network.species[1]}')
        plt.xlabel(f'Concentration [mass fr.] {self.reaction_network.species[0]}')
        if save_file is not None:
            plt.savefig(f"{save_file}_{self.i}.png")
        plt.show()
    
    def plot_result_2d_x_in_3d(self,save_file=None):
        """
        plot the result when the reaction has 2 species with fixed reaction rates
        """
        ax = plt.figure().add_subplot(projection='3d')
        x_plot = self.x_plot.view(-1,2)
        x_plot_model = self.x_plot_model.view(-1,2)
        x_data = self.x_training_data.view(-1,2)
        x_data_model = self.x_training_data_model.view(-1,2)
        print(self.t_plot.shape, x_plot[:,0].shape, x_plot[:,1].shape)
        ax.plot(self.t_plot.numpy(), x_plot[:,0].numpy(),x_plot[:,1].numpy(), color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        ax.plot(self.t_plot.numpy(), x_plot_model[:,0].numpy(),x_plot_model[:,1].numpy(), color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
        ax.scatter(self.t_training_data.numpy(), x_data[:,0].numpy(), x_data[:,1].numpy(), s=60, color="tab:orange", alpha=0.4, label='Training data')
        ax.scatter(self.t_training_data.numpy(), x_data_model[:,0].numpy(), x_data_model[:,1].numpy(), s=60, color="yellow", alpha=0.4, label='Training data prediction')
        l = ax.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        ax.set_zlabel(f'Concentration [mass fr.] {self.reaction_network.species[1]}')
        ax.set_ylabel(f'Concentration [mass fr.] {self.reaction_network.species[0]}')
        ax.set_xlabel('Time [s]')
        if save_file is not None:
            plt.savefig(f"{save_file}_{self.i}.png")
        plt.show()
    
    def plot_losses(self,save_file=None):
        """
        Plot the total, data and physics losses with respect to the training epochs
        """
        plt.figure(figsize=(10,5))
        plt.plot(range(self.epoch), self.total_losses[:self.epoch], color="grey", linewidth=2, label="Total loss")
        plt.plot(range(self.epoch), self.data_losses[:self.epoch], color="tab:blue", linewidth=2, label="Data loss")
        plt.plot(range(self.epoch), self.physics_losses[:self.epoch], color="tab:orange", linewidth=2, label="Physics loss")
        l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        plt.xlabel(f'Epoch')
        plt.ylabel(f'Loss')
        if save_file is not None:
            plt.savefig(f"{save_file}_{self.i}.png")
        plt.show()

    def plot_species_vs_time_at_params(self, species: str, parameter_index: tuple, show_physics=True,save_file=None):
        """
        Plot the concentration/time of the given species at reaction rate given by `parameter_index`,
        the index into the reaction network parameter meshgrid self.reaction_network.reaction_rates.meshgrid
        """
        
        ind = (slice(None),) + parameter_index + (self.reaction_network.species_index[species],)
        
        assert self.reaction_network.reaction_rates.meshgrid.dim() == len(parameter_index) + 1, f"parameter_index must have length {self.reaction_network.reaction_rates.meshgrid.dim()-1} but it has length {len(parameter_index)}"
        
        plt.figure(figsize=(8,4))
        plt.plot(self.t_plot,self.x_plot[ind], color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        plt.plot(self.t_plot,self.x_plot_model[ind], color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
        plt.scatter(self.t_training_data, self.x_training_data[ind], s=60, color="tab:orange", alpha=0.4, label='Training data')
        if show_physics:
            plt.scatter(self.t_training_physics, self.x_training_physics[ind], s=60, color="tab:green", alpha=0.4, 
                        label='NN kinetic DE rhs')
            plt.scatter(self.t_training_physics, self.x_training_physics_model[ind], s=60, color="tab:red", alpha=0.4, 
                        label='NN dxdt')
        l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        plt.ylabel(f'Concentration of {species} [mass fr.]')
        plt.xlabel('Time [s]')
        if save_file is not None:
            plt.savefig(f"{save_file}_{self.i}.png")
        plt.show()
    
def train_rn_model(
    model,
    rn: ReactionNetwork,
    x0, t_data,
    t_physics, epochs=10000,
    t_plot = None,
    seed = 123,
    optimizer = None,
    total_loss = lambda data, physics: data+0.3*physics,
    plot_data_config=None,
):
    """
    Parameters:
        model:      the NN we want to train as a PINN.
                        number of inputs:  1 + rn.n_param_reaction_rate() (t + input k-s)
                        number of outputs: rn.species()                   (concentration of species)
        rn:         the reaction network
        x0:         tensor, initial concentration
        t_data:     tensor, the points where the model is fit to the real, numeric solution of the kinetic de of rn, species concentrations
        t_physics:  tensor, the points where the derivative of the model is fit to the kinetic de of rn
        epochs:     the number of training epochs
        t_plot:     tensor, times where the real points and model predictions should be plotted. Default is linspace(min(t_data, t_physics), max(t_data, t_physics), 500)
        seed:       seed used for torch.manual_seed
        optimizer:  optimizer used to train the model, default is torch.optim.Adam(model.parameters(),lr=1e-4)
        total_loss: total_loss(data_loss, physics_loss) calculates the total loss from the two losses. data+0.3*physics
    """
    assert(x0.dtype == torch.float32)
    assert(t_data.dtype == torch.float32)
    assert(t_physics.dtype == torch.float32)
    assert(rn.n_species() == x0.shape[0])
    t_physics.requires_grad_(True)
    if t_plot is None:
        t_plot = torch.linspace(min(float(t_data.min()), t_physics.min().item()), max(t_data.max().item(), t_physics.max().item()), 500).view(-1,1)
    else:
        t_plot = t_plot.view(-1,1)
    assert(t_plot.dtype == torch.float32)
    torch.manual_seed(seed)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    t_data = t_data.view(-1,1)
    t_physics = t_physics.view(-1,1)
    input_physics = rn.reaction_rates.input_meshgrid(t_physics.view(-1))
    input_data = rn.reaction_rates.input_meshgrid(t_data.view(-1))
    input_plot = rn.reaction_rates.input_meshgrid(t_plot.view(-1))
    x_data = torch.tensor(rn.numeric_solve_mass_action_kinetics_de(x0, max(t_data), t_eval=t_data.view(-1)), dtype=torch.float32)
    x_plot = torch.tensor(rn.numeric_solve_mass_action_kinetics_de(x0, max(t_plot), t_eval=t_plot.view(-1)), dtype=torch.float32)
    
    total_losses = []
    data_losses = []
    physics_losses = []
    for i in range(epochs):
        optimizer.zero_grad()
        x_data_pred = model(input_data)
        physics_loss = rn.mass_action_kinetics_loss(model, input_physics, t_physics)
        data_loss = mse_loss(x_data, x_data_pred.view(x_data.shape))
        kinetics_loss = rn.mass_action_kinetics_loss(model, input_physics, t_physics)
        total_loss_ = total_loss(data_loss, kinetics_loss)
        total_loss_.backward()
        optimizer.step()
        
        total_losses.append(total_loss_.item())
        physics_losses.append(physics_loss.item())
        data_losses.append(data_loss.item())
        
        # plot the result as training progresses
        if plot_data_config is not None and (i+1) % plot_data_config.plot_interval == 0:
            yield PlotData(rn, i+1, (i+1)/plot_data_config.plot_interval,
                           total_losses, data_losses, physics_losses,
                           t_plot, model(input_plot).detach(), x_plot,
                           t_data, model(input_data).detach(), x_data,
                           t_physics.detach(), rn.dxdt.detach(), rn.dxdt_mass_action.detach())

