#!/usr/bin/env python3
# pinn_hyperparam_search_bayesian_with_datafix.py

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import random

# For parallel random search
from joblib import Parallel, delayed

# For Bayesian optimisation
import optuna

###############################################################################
#                               Data loading
###############################################################################
def load_data(path='Plate_data.mat'):
    """
    Loads data as double precision (float64).
    """
    data = scipy.io.loadmat(path)
    
    # We'll keep everything in double precision.
    L_boundary = torch.tensor(data['L_boundary'], dtype=torch.float64)
    R_boundary = torch.tensor(data['R_boundary'], dtype=torch.float64)
    T_boundary = torch.tensor(data['T_boundary'], dtype=torch.float64)
    B_boundary = torch.tensor(data['B_boundary'], dtype=torch.float64)
    C_boundary = torch.tensor(data['C_boundary'], dtype=torch.float64)
    Boundary   = torch.tensor(data['Boundary'], dtype=torch.float64, requires_grad=True)

    disp_truth = torch.tensor(data['disp_data'], dtype=torch.float64)
    t_connect  = torch.tensor(data['t'].astype(float), dtype=torch.float64)

    x_full = torch.tensor(data['p_full'], dtype=torch.float64, requires_grad=True)
    x      = torch.tensor(data['p'],      dtype=torch.float64, requires_grad=True)

    # Example: pick 50 random points for "data_loss_fix" from part (e)
    rand_index = torch.randint(0, len(x_full), (50,))
    disp_fix = disp_truth[rand_index,:]

    return (L_boundary, R_boundary, T_boundary, B_boundary, C_boundary,
            Boundary, disp_truth, t_connect, x_full, x,
            rand_index, disp_fix)

###############################################################################
#                          Network architecture
###############################################################################
class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1

        modules = []
        for j in range(self.n_layers):
            modules.append(nn.Linear(layers[j], layers[j+1]))
            # Activation for all but the last layer
            if j != self.n_layers - 1:
                modules.append(nonlinearity())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

def build_networks(N_layer_hidden_disp, N_node_hidden_disp,
                   N_layer_hidden_stress, N_node_hidden_stress):
    """
    Build the displacement and stress networks, by default in float (we'll convert to double).
    """
    # Architecture of displacement net
    Disp_layer = [2]  # input dimension (x,y)
    for _ in range(N_layer_hidden_disp):
        Disp_layer.append(N_node_hidden_disp)
    Disp_layer.append(2)  # output dimension (u, v)

    # Architecture of stress net
    Stress_layer = [2]  # input dimension (x,y)
    for _ in range(N_layer_hidden_stress):
        Stress_layer.append(N_node_hidden_stress)
    Stress_layer.append(3)  # output dimension (sigma11, sigma22, sigma12)

    disp_net   = DenseNet(Disp_layer, nn.Tanh)
    stress_net = DenseNet(Stress_layer, nn.Tanh)
    return disp_net, stress_net

###############################################################################
#                           Training function
###############################################################################
def train_pinn(learning_rate,
               step_size,
               decay_factor,
               max_epochs,
               N_layer_hidden_disp,
               N_node_hidden_disp,
               N_layer_hidden_stress,
               N_node_hidden_stress,
               stiff_template,
               x, Boundary,
               L_boundary, R_boundary, T_boundary, B_boundary, C_boundary,
               x_full=None, disp_fix=None, rand_index=None,
               # We now ALWAYS apply the data-fix lines from your snippet
               early_stopping=False,
               early_stopping_patience=5000,
               verbose=True):
    """
    Trains a PINN for a specified number of epochs using the given hyperparameters,
    returning final loss and references to the trained networks.
    Incorporates the 'data loss fix' lines directly, as in lines ~225-233 of your snippet.
    """

    # 1. Build fresh networks
    disp_net, stress_net = build_networks(
        N_layer_hidden_disp, N_node_hidden_disp,
        N_layer_hidden_stress, N_node_hidden_stress
    )
    # Convert them to double
    disp_net = disp_net.double()
    stress_net = stress_net.double()

    # 2. Optimizer and scheduler
    params = list(disp_net.parameters()) + list(stress_net.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=step_size,
                                                gamma=decay_factor)
    loss_func = nn.MSELoss()

    # 3. Broadcast stiffness
    stiff      = torch.broadcast_to(stiff_template, (len(x),        3, 3))
    stiff_bc   = torch.broadcast_to(stiff_template, (len(Boundary), 3, 3))

    best_loss = 1e30
    epochs_since_improvement = 0

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # =============== Forward pass for internal points ===============
        sigma = stress_net(x)  # predicted stress
        disp  = disp_net(x)    # predicted displacement
        u = disp[:, 0]
        v = disp[:, 1]

        # Derivatives for strain
        dudx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        dvdx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                   create_graph=True)[0]

        e_11 = dudx[:,0].unsqueeze(1)
        e_22 = dvdx[:,1].unsqueeze(1)
        e_12 = 0.5*(dudx[:,1] + dvdx[:,0]).unsqueeze(1)
        e    = torch.cat((e_11, e_22, e_12), dim=1).unsqueeze(2)

        # Augmented stress
        sig_aug = torch.bmm(stiff, e).squeeze(2)
        loss_cons = loss_func(sig_aug, sigma)

        # Boundary version
        disp_bc  = disp_net(Boundary)
        sigma_bc = stress_net(Boundary)
        u_bc = disp_bc[:,0]
        v_bc = disp_bc[:,1]

        dudx_bc = torch.autograd.grad(u_bc, Boundary,
                                      grad_outputs=torch.ones_like(u_bc),
                                      create_graph=True)[0]
        dvdx_bc = torch.autograd.grad(v_bc, Boundary,
                                      grad_outputs=torch.ones_like(v_bc),
                                      create_graph=True)[0]

        e_11_bc = dudx_bc[:,0].unsqueeze(1)
        e_22_bc = dvdx_bc[:,1].unsqueeze(1)
        e_12_bc = 0.5*(dudx_bc[:,1] + dvdx_bc[:,0]).unsqueeze(1)
        e_bc = torch.cat((e_11_bc, e_22_bc, e_12_bc), 1).unsqueeze(2)

        sig_aug_bc  = torch.bmm(stiff_bc, e_bc).squeeze(2)
        loss_cons_bc = loss_func(sig_aug_bc, sigma_bc)

        # Equilibrium
        sig_11 = sigma[:,0]
        sig_22 = sigma[:,1]
        sig_12 = sigma[:,2]

        dsig11dx = torch.autograd.grad(sig_11, x,
                                       grad_outputs=torch.ones_like(sig_11),
                                       create_graph=True)[0]
        dsig22dx = torch.autograd.grad(sig_22, x,
                                       grad_outputs=torch.ones_like(sig_22),
                                       create_graph=True)[0]
        dsig12dx = torch.autograd.grad(sig_12, x,
                                       grad_outputs=torch.ones_like(sig_12),
                                       create_graph=True)[0]

        eq_x1 = dsig11dx[:,0] + dsig12dx[:,1]
        eq_x2 = dsig12dx[:,0] + dsig22dx[:,1]

        # zero body forces
        f_x1 = torch.zeros_like(eq_x1)
        f_x2 = torch.zeros_like(eq_x2)
        loss_eq1 = loss_func(eq_x1, f_x1)
        loss_eq2 = loss_func(eq_x2, f_x2)

        # Boundary conditions
        tau_R = 0.1
        tau_T = 0

        u_L   = disp_net(L_boundary)
        u_B   = disp_net(B_boundary)
        sig_R = stress_net(R_boundary)
        sig_T = stress_net(T_boundary)
        sig_C = stress_net(C_boundary)

        # symmetry BC left
        loss_BC_L = loss_func(u_L[:,0], torch.zeros_like(u_L[:,0]))
        # symmetry BC bottom
        loss_BC_B = loss_func(u_B[:,1], torch.zeros_like(u_B[:,1]))
        # traction BC right
        loss_BC_R = (loss_func(sig_R[:,0], tau_R*torch.ones_like(sig_R[:,0]))
                     + loss_func(sig_R[:,2], torch.zeros_like(sig_R[:,2])))
        # traction BC top
        loss_BC_T = (loss_func(sig_T[:,1], tau_T*torch.ones_like(sig_T[:,1]))
                     + loss_func(sig_T[:,2], torch.zeros_like(sig_T[:,2])))
        # traction-free on circle
        loss_BC_C = (loss_func(sig_C[:,0]*C_boundary[:,0] + sig_C[:,2]*C_boundary[:,1],
                               torch.zeros_like(sig_C[:,0]))
                     + loss_func(sig_C[:,2]*C_boundary[:,0] + sig_C[:,1]*C_boundary[:,1],
                                 torch.zeros_like(sig_C[:,0])))

        # ===================== NEW LINES (Part e) =====================
        # data_loss_fix, lines ~225-233 from your snippet
        x_fix   = x_full[rand_index, :]
        u_fix   = disp_net(x_fix)
        loss_fix = loss_func(u_fix, disp_fix)

        # Combine everything, including part(e) with a weight of 100
        loss = (loss_eq1 + loss_eq2
                + loss_cons + loss_cons_bc
                + loss_BC_L + loss_BC_B + loss_BC_R + loss_BC_T + loss_BC_C
                + 100.0 * loss_fix)
        # =============================================================

        # Backprop + step
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step the LR scheduler after optimizer

        current_loss = float(loss.detach().cpu().numpy())

        # Early stopping
        if current_loss < best_loss:
            best_loss = current_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if early_stopping and epochs_since_improvement > early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch={epoch}, no improvement for {early_stopping_patience} epochs.")
            break

        # Print progress occasionally
        if verbose and (epoch % 1000 == 0):
            print(f"[{epoch}/{max_epochs}] loss = {current_loss:.6f}")

    return best_loss, disp_net, stress_net

###############################################################################
#                    Multi-stage: Random + Bayesian Optimisation
###############################################################################
def multi_stage_search():
    """
    Demonstration of:
      1) Random search to generate N candidates.
      2) Keep top K, feed them into a Bayesian (Optuna) search.
      3) Now includes the data-loss-fix lines in the training loop.
      4) Print final best hyperparameters (no final 50k training).
    """

    #-------------------- 1. Load data, define stiffness --------------------#
    (L_boundary, R_boundary, T_boundary, B_boundary, C_boundary,
     Boundary, disp_truth, t_connect, x_full, x,
     rand_index, disp_fix) = load_data(path='Plate_data.mat')

    # Example stiffness for plane stress
    E = 10
    mu = 0.3
    stiff_template = E/(1-mu**2)*torch.tensor([[1, mu, 0],
                                              [mu, 1, 0],
                                              [0,  0, (1-mu)/2]], dtype=torch.float64)
    stiff_template = stiff_template.unsqueeze(0)

    def short_train(hparams, epochs=400):
        best_loss, _, _ = train_pinn(
            learning_rate=hparams['lr'],
            step_size=10000,
            decay_factor=0.5,
            max_epochs=epochs,
            N_layer_hidden_disp=hparams['disp_layers'],
            N_node_hidden_disp=hparams['disp_nodes'],
            N_layer_hidden_stress=hparams['stress_layers'],
            N_node_hidden_stress=hparams['stress_nodes'],
            stiff_template=stiff_template,
            x=x, Boundary=Boundary,
            L_boundary=L_boundary, R_boundary=R_boundary, T_boundary=T_boundary,
            B_boundary=B_boundary, C_boundary=C_boundary,
            x_full=x_full, disp_fix=disp_fix, rand_index=rand_index,
            early_stopping=False,
            verbose=False
        )
        return best_loss

    #-------------------- 2. Random search stage 1 --------------------#
    N_candidates = 6
    possible_layers = [2, 3]            
    possible_nodes  = [6, 128, 256, 400]
    random_hparam_sets = []

    for _ in range(N_candidates):
        hparams = {
            'disp_layers': random.choice(possible_layers),
            'stress_layers': random.choice(possible_layers),
            'disp_nodes': random.choice(possible_nodes),
            'stress_nodes': random.choice(possible_nodes),
            'lr': 10**(random.uniform(-4, -2)),
        }
        random_hparam_sets.append(hparams)

    print(f"=== Stage 1: Random search over {N_candidates} configs, each short-trained (2000 epochs) ===")
    results_stage1 = Parallel(n_jobs=2)(
        delayed(short_train)(hp) for hp in random_hparam_sets
    )

    stage1_with_loss = [(loss_val, hp) for loss_val, hp in zip(results_stage1, random_hparam_sets)]
    stage1_with_loss.sort(key=lambda x: x[0])

    K = 2
    topK_stage1 = stage1_with_loss[:K]

    print("\nTop K from Stage 1 (random search):")
    for rank, (loss_val, hp) in enumerate(topK_stage1):
        print(f" Rank {rank+1}, loss={loss_val:.6f}, hparams={hp}")

    #-------------------- 3. Bayesian (Optuna) stage 2 --------------------#
    M = 5
    bayes_epochs = 5000

    def optuna_objective(trial: optuna.Trial):
        # If you still want to choose between 2 or 3 layers:
        disp_layers = trial.suggest_categorical("disp_layers", [2, 3])
        stress_layers = trial.suggest_categorical("stress_layers", [2, 3])
        
        # Now pick # of nodes in [6, 400] as an integer, uniformly:
        disp_nodes = trial.suggest_int("disp_nodes", 6, 400)
        stress_nodes = trial.suggest_int("stress_nodes", 6, 400)
        
        # For LR, pick a continuous value in [1e-4, 1e-2] on a log scale:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        # The rest is as before:
        hparams = {
            'disp_layers': disp_layers,
            'stress_layers': stress_layers,
            'disp_nodes': disp_nodes,
            'stress_nodes': stress_nodes,
            'lr': lr
        }
    
        # Train for e.g. 5k epochs:
        loss_val = short_train(hparams, epochs=bayes_epochs)
        return loss_val

    print(f"\n=== Stage 2: Bayesian search with Optuna (initial points = top {K} from stage 1, plus {M} new) ===")

    study = optuna.create_study(direction="minimize")

    for (loss_val, hp) in topK_stage1:
        fixed_params = {
            "disp_layers": hp["disp_layers"],
            "stress_layers": hp["stress_layers"],
            "disp_nodes": hp["disp_nodes"],
            "stress_nodes": hp["stress_nodes"],
            "lr": hp["lr"],
        }
        study.enqueue_trial(fixed_params)

    study.optimize(optuna_objective, n_trials=K + M)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_loss_val = best_trial.value

    print("\n=== Best trial from Stage 2 (Optuna) ===")
    print(f"Hyperparameters: {best_params}")
    print(f"Loss = {best_loss_val:.6f}")

    print("\n======= Final best set of hyperparameters: =======")
    print(best_params)
    print(f"Associated short-run loss = {best_loss_val:.6f}")
    print("\nNo final 50k training is performed here.")


###############################################################################
#                         Main entry point
###############################################################################
if __name__ == "__main__":
    # For reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    multi_stage_search()
