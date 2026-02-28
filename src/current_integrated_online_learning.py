"""Current integrated online learning PINN training script.

Generated from Current Integrted_online_learning (another copy).ipynb.
"""

# %%
######################## Imports ########################
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
#from ipywidgets import interact, IntSlider

# %%
DIMENSIONS = 2 # number of dimensions of the PDE
BOUNDARIES = [
                [0.0, .40],  # Time boundaries
                [0.0, 1],  # x boundaries
             #charging current boundaries
                #[0.0, 1.0],  # y boundaries
                # Add more boundaries for additional dimensions if needed
            ]
####################################################################################



######################## NEURAL NETWORK ########################
NUMBER_HIDDEN_LAYERS = 2 # number of hidden layers in the neural network
NUMBER_NEURONS_PER_LAYER = 20 # number of neurons per hidden layer
ACTIVATION = tf.keras.activations.tanh # activation function
##################################################################



######################## DATA POINTS ########################
NUMBER_DATA_POINTS_PDE = 2500 # number of random data points
NUMBER_DATA_POINTS_INITIAL_CONDITION = 750 # number of random initial condition data points
NUMBER_DATA_POINTS_BOUNDARY_CONDITION = 1200 # number of random boundary condition data points
##################################################################



######################## TRAINING ########################
NUMBER_TRAINING_EPOCHS = 154000 # number of training iterations
PRINT_LOSS_INTERVAL = 10 # print the loss every PRINT_LOSS_INTERVAL training steps
PLOT_LOSS_HISTORY = True # boolean to plot or not plot the loss history after training
##################################################################



######################## SAVING THE MODEL ########################
CHECKPOINT_PATH = 'model_checkpoint' #path to save the model
CHECKPOINT_ITERATIONS = 10000 # save the model each CHECKPOINT_ITERATIONS iterations
##################################################################



######################## DATA TYPE ########################
DTYPE='float32' # Set data type
tf.keras.backend.set_floatx(DTYPE)
###########################################################



######################## PARAMETERS ########################
pi = tf.constant(np.pi)
D = .8e-16 # diffusion coeff
r = 1 # growth rate
############################################################



######################## Define initial condition ########################
def comp_i(X):
    x = X[:,1:2]
    y = X[:,2:3]
    return 0.04



######################## Define the boundary condition ########################
CHOOSE_NEUMANN_BOUNDARY_CONDITION = True # if true->Neumann BC,   if false->Dirichlet BC

######################## NEUMANN BOUNDARY CONDITION ########################
def neuman_condition(I):
    NEUMANN_BOUNDARY_CONDITION =-0.197*I# the Neumann BC
    return NEUMANN_BOUNDARY_CONDITION



######################## DIRICHLET BOUNDARY CONDITION ########################
def DC_comp_b(prediction, x_b):

    # variables
    t = x_b[:,0:1]
    x = x_b[:,1:2]
    y = x_b[:,2:3]

    expected = tf.sin(pi*x)+tf.sin(pi*y) # expected output at the boundary

    return prediction - expected 




######################## Define residual of the PDE ########################
def comp_r(var, u, first_deriv, second_deriv):
    # Extracting first derivatives
    u_t = first_deriv[0]
    u_x = first_deriv[1]

    # Extracting the required second derivative
    u_xx = second_deriv[1]

    # Compute the PDE residual
    residual = u_t - (2/ var[1]) * u_x - u_xx

    return residual





######################## YOU DON'T NEED TO MODIFY THIS FUNCTION ########################
######################## residual of neumann boundary condition ########################
def NBC_comp_b(var, first_derivatives, lower_boundary, upper_boundary,current_values):

    neumann_boundary_condition = neuman_condition(current_values) # get the neumann boundary condition

    return_tensor = tf.zeros_like(var[1]) # init the return tensor

    for i in range(1,len(var)): # we start from 1 to skip the time variable
        comparison_tensor = tf.equal(var[i], upper_boundary[i]) # is our tensor on the boundaries?
        return_tensor += tf.where(
                                    comparison_tensor, 
                                    [first_derivatives[i]-neumann_boundary_condition,first_derivatives[i]-neumann_boundary_condition], 
                                    0
                                ) # if it is in the boundary then return the first derivative of that variable minus the neumann BC, if not 0

    return return_tensor
import tensorflow as tf

def apply_neumann_boundary_condition(var, first_derivatives, lower_boundary, upper_boundary):
    # Neumann boundary condition value: derivative is zero at x=0
    neumann_boundary_value = 0.0

    # Initialize a tensor to store the boundary condition application
    neumann_bc_application = tf.zeros_like(var[1])  # Assuming var[1] is 'x'

    # Loop over the variables, starting from 1 to skip the time variable
    for i in range(1, len(var)):
        # Apply the boundary condition only to the x variable
        if i == 1:  # Assuming var[1] is 'x'
            # Check if x is at its lower boundary (i.e., x=0)
            is_at_lower_boundary = tf.equal(var[i], lower_boundary[i])

            # Apply the boundary condition: (first_derivative - neumann_boundary_value) at x=0
            neumann_bc_application += tf.where(
                is_at_lower_boundary, 
                first_derivatives[i] - neumann_boundary_value, 
                0.0
            )

    return neumann_bc_application

# Example usage
# var: tensor containing variables (e.g., [t, x])
# first_derivatives: tensor containing the first derivatives with respect to each variable
# lower_boundary and upper_boundary: tensors containing the lower and upper boundary values for each variable

# %%
N_0 = NUMBER_DATA_POINTS_INITIAL_CONDITION  # Number of initial condition data points
N_b = NUMBER_DATA_POINTS_BOUNDARY_CONDITION  # Number of boundary data points
N_r = NUMBER_DATA_POINTS_PDE  # Number of PDE data points

# Define the boundaries for each dimension
boundaries = BOUNDARIES

# Set the lower and upper bounds as tensors
lb = tf.constant([boundaries[i][0] for i in range(len(boundaries))], dtype=DTYPE)  # Lower bounds
ub = tf.constant([boundaries[i][1] for i in range(len(boundaries))], dtype=DTYPE)  # Upper bounds


################ Set random seed for reproducible results or remove it for different results ################
tf.random.set_seed(0)



################################ Initial data ################################
t_0 = tf.ones((N_0, 1), dtype=DTYPE) * lb[0]  # Time values for initial condition
other_0 = tf.random.uniform((N_0, DIMENSIONS-1), lb[1:], ub[1:], dtype=DTYPE)  # Random spatial values for initial condition
#other_0=tf.ones((N_0, 1), dtype=DTYPE) * lb[0] 
X_0 = tf.concat([t_0, other_0], axis=1) # Initial Condtition points





################################ Boundary data ################################
t_b = tf.linspace(lb[0], ub[0], N_b)

# Generate other boundary values; here it's constant at ub[1]
other_b = tf.ones((N_b, 1), dtype=tf.float32) * ub[1]

# Concatenate time and other boundary values
X_b = tf.concat([tf.expand_dims(t_b, axis=1), other_b], axis=1)
other_b_n=tf.ones((N_b, 1), dtype=DTYPE) * lb[1]
X_b_n=tf.concat([tf.expand_dims(t_b, axis=1),other_b_n],axis=1)

print(X_b)

#other_b_n=tf.ones((N_0, 1), dtype=DTYPE) * lb[0]
#X_b_n=tf.concat([t_b,other_b_n],axis=1)

# if we are in 2 DIMENSIONS we already have the BC points with the lines above






################################ Draw uniformly sampled collocation points ################################
 # Random time and spatial values for PDE data
# Generate the time values (equivalent to torch.arange and unsqueeze)
import tensorflow as tf

# Define bounds and number of random samples

# Generate the time values (equivalent to torch.arange and unsqueeze)
t_r = tf.linspace(lb[0], ub[0], N_r)
t_r = tf.reshape(t_r, [-1, 1])  # reshaping to (N_r, 1)

# Generate the random spatial values, excluding boundary values (lb[1], ub[1])
x_r = tf.random.uniform([N_r, 1], minval=lb[1] + 1e-2, maxval=ub[1] - 1e-2)

# Concatenate time and spatial values
X_r = tf.concat([t_r, x_r], axis=1)

# View the result

# %%
import tensorflow as tf
import matplotlib.pyplot as plt

# Define bounds and number of time points
 # number of time points

# Generate the time points
time_points = tf.linspace(lb[0], ub[0], N_b)

# Calculate the quarter time index
quarter_time = tf.cast(N_b / 4, tf.int32)

# Create the condition for the first 25% of the time points
first_25_percent = tf.less_equal(tf.range(N_b), quarter_time)

# Define the decay rate and start time for decay
decay_rate = 8
decay_start_time = time_points[quarter_time:]

# Calculate the decaying function for the remaining 75% of the time
decay_values = -4* tf.exp(-decay_rate * (time_points[quarter_time:] - time_points[quarter_time]) / (ub[0] - time_points[quarter_time]))

# Set the last 7% of the time to zero
last_7_percent_index = tf.cast(0.93 * N_b, tf.int32)
last_7_percent = tf.greater_equal(tf.range(N_b), last_7_percent_index)

# Create the final current values, applying the conditions
current_values = tf.where(
    first_25_percent, 
    -4.0, 
    tf.concat([tf.zeros_like(time_points[:quarter_time]), decay_values], axis=0)
)

# Set the last 7% to zero
current_values = tf.where(last_7_percent, 0.0, current_values)

# Plot the result
plt.plot(time_points, current_values)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.title('Current vs Time (0 to 4 seconds) with Last 7% Zero')
plt.grid(True)
plt.show()

# %%
current_values=tf.cast(current_values, tf.float32)
current_values.dtype

# %%
current_values.shape

# %%
X_b.shape

# %%
import tensorflow as tf

def init_model(num_hidden_layers=NUMBER_HIDDEN_LAYERS, 
               num_neurons_per_layer=NUMBER_NEURONS_PER_LAYER, 
               current_values=None):
    # Input: x and I
    input_var = tf.keras.Input(shape=(2,))  # input_var with shape (100, 2)
    
    # Check if current_values is provided
    if current_values is not None:
        input_values = tf.keras.Input(shape=(1,))  # current_values with shape (100, 1)
        
        # Scaling layer to map [lb, ub] to [0,1] (Assume lb and ub are predefined)
        scaling_layer = tf.keras.layers.Lambda(lambda x: (x - lb) / (ub - lb))(input_var)
        
        # Concatenate inputs
        concatenated_inputs = tf.keras.layers.Concatenate()([scaling_layer, input_values])
        
        x = concatenated_inputs
        print(x)
    else:
        # Scaling layer to map [lb, ub] to [0,1] if no current_values
        #x = tf.keras.layers.Lambda(lambda x: (x - lb) / (ub - lb))(input_var)
        x=input_var
    
    # Add hidden layers
    for _ in range(num_hidden_layers):
        x = tf.keras.layers.Dense(num_neurons_per_layer,
                                  activation=ACTIVATION,
                                  kernel_initializer='glorot_normal')(x)

    # Output layer
    output = tf.keras.layers.Dense(1)(x)

    if current_values is not None:
        # Define the model with multiple inputs if current_values is provided
        model = tf.keras.Model(inputs=[input_var, input_values], outputs=output)
    else:
        # Define the model with only input_var if current_values is not provided
        model = tf.keras.Model(inputs=input_var, outputs=output)

    return model

# %%
model = init_model()
current_values_expand=tf.expand_dims(current_values, axis=1)
current_values_expand.shape

# %%
X_b

# %%
tf.concat([X_b, current_values_expand], axis=1)

# %%
model(X_b)

# %%
def NBC_comp_b(var, first_derivatives, lower_boundary, upper_boundary,current_values):

    neumann_boundary_condition = neuman_condition(current_values) # get the neumann boundary condition

    return_tensor = tf.zeros_like(var[1]) # init the return tensor

    for i in range(1,len(var)): # we start from 1 to skip the time variable
        comparison_tensor = tf.equal(var[i], upper_boundary[i]) # is our tensor on the boundaries?
        return_tensor += tf.where(
                                    comparison_tensor, 
                                    [first_derivatives[i]-neumann_boundary_condition,first_derivatives[i]-neumann_boundary_condition], 
                                    0
                                ) # if it is in the boundary then return the first derivative of that variable minus the neumann BC, if not 0

    return return_tensor
import tensorflow as tf

# %%
variables = []
for i in range(DIMENSIONS):
    variables.append(X_b_n[:,i:i+1])





stack = [] # stack the variables to feed the model
for i in range(DIMENSIONS):
    stack.append(variables[i][:,0])

# %%
stack[0].shape

# %%
tf.stack(stack, axis=1)

# %%
num_points = 1200

# Generate t linearly spaced between 0 and ln(2) to ensure (1 - e^(-t)) ranges from 0 to 0.5
t = tf.linspace(0.0, 0.5, num_points)

# Compute the discount factor (1 - e^(-t)) which will range from 0 to 0.5
discount_factor =  tf.math.exp(-10*t)
discount_factor

# %%
################ Get residual of PDE ################
def get_r(model, X_r,current_values):
    #current_values_expand=tf.expand_dims(current_values, axis=1)
    # Compute derivatives
    with tf.GradientTape(persistent=True) as tape: # A tf.GradientTape is used to compute derivatives in TensorFlow

        # split variables to compute partial derivatives
        variables = []
        for i in range(DIMENSIONS):
            variables.append(X_r[:,i:i+1])
    
        
        # watch the variables
        for i in range(DIMENSIONS):
            tape.watch(variables[i])
        
        
        stack = [] #stack the variables to feed the model
        for i in range(DIMENSIONS):
            stack.append(variables[i][:,0]) # add to list to stack, drop the second dimension

        # Determine residual 
        u = model((tf.stack(stack, axis=1)))
        
        # Compute the first gradients
        first_derivatives = [] 
        for i in range(DIMENSIONS): # we compute this derivatives within the gradient tape because we need the second derivatives 
            first_derivatives.append(tape.gradient(u, variables[i])) 


    # Compute the second derivatives 
    second_derivatives = [] # choose which second derivatives you want, there are on the order [x,y,z,...]->[u_xx,u_yy,u_zz,...]
    for i in range(DIMENSIONS):
        second_derivatives.append(tape.gradient(first_derivatives[i], variables[i]))

    del tape

    return comp_r(variables, u, first_derivatives, second_derivatives)




################ Get residual of boundary condition ################
def get_b_r(model ,x_b,current_values,discount_factor):
    #current_values_expand=tf.expand_dims(current_values, axis=1)
    if CHOOSE_NEUMANN_BOUNDARY_CONDITION==False: # if we have Dirichlet Boundary Condition
        prediction = model(x_b)
        return DC_comp_b(prediction,x_b)
    
    else:
        # Compute the derivatives
        with tf.GradientTape(persistent=True) as tape: # A tf.GradientTape is used to compute derivatives in TensorFlow
            
            # split variables to compute partial derivatives
            variables = []
            for i in range(DIMENSIONS):
                variables.append(x_b[:,i:i+1])

            # watch variables
            for i in range(DIMENSIONS):
                tape.watch(variables[i])

            
            
            stack = [] # stack the variables to feed the model
            for i in range(DIMENSIONS):
                stack.append(variables[i][:,0]) # add to list to stack, drop the second dimension
            
            # Determine residual
            u =model((tf.stack(stack, axis=1)))
            
        # Compute the first gradients
        first_derivatives = [0] # remove the 0 and start the loop from 0 if you also need the time derivative
        for i in range(1,DIMENSIONS): # we skip the derivative of time since we don't need it for boundary contitions
            first_derivatives.append(tape.gradient(u, variables[i]))


        del tape


        return NBC_comp_b(variables,first_derivatives,lb,ub,current_values)*discount_factor
def get_b_r_n(model,x_b_n,current_values,discount_factor):
    #current_values_expand=tf.expand_dims(current_values, axis=1)
    with tf.GradientTape(persistent=True) as tape: # A tf.GradientTape is used to compute derivatives in TensorFlow
            
            # split variables to compute partial derivatives
            variables = []
            for i in range(DIMENSIONS):
                variables.append(x_b_n[:,i:i+1])

            # watch variables
            for i in range(DIMENSIONS):
                tape.watch(variables[i])

            
            
            stack = [] # stack the variables to feed the model
            for i in range(DIMENSIONS):
                stack.append(variables[i][:,0]) # add to list to stack, drop the second dimension
            
            # Determine residual
            u = model((tf.stack(stack, axis=1)))
            
        # Compute the first gradients
    first_derivatives = [0] # remove the 0 and start the loop from 0 if you also need the time derivative
    for i in range(1,DIMENSIONS): # we skip the derivative of time since we don't need it for boundary contitions
        first_derivatives.append(tape.gradient(u, variables[i]))


        del tape
    
    return apply_neumann_boundary_condition(variables,first_derivatives,lb,ub)*discount_factor




################ Get residual of initial condition ################
def get_i_r(model, x_0,current_values):
    #current_values_expand=tf.expand_dims(current_values, axis=1)
    return comp_i(x_0) - model(x_0)




################ Get loss ################
def compute_loss(model, x_r, x_b, x_0,X_b_n,current_values,discount_factor):

    # Compute phi^r
    r = get_r(model, x_r,current_values)
    phi_r = tf.reduce_mean(tf.square(r))

    # compute phi^b
    b = get_b_r(model, x_b,current_values,discount_factor)
    phi_b = tf.reduce_mean(tf.square(b))

    # Compute phi^0
    i = get_i_r(model, x_0,current_values)
    phi_0 = tf.reduce_mean(tf.square(i))

    b_n=get_b_r_n(model,X_b_n,current_values,discount_factor)
    phi_b_n=tf.reduce_mean(tf.square(b_n))
    # Initialize loss
    loss = phi_r +(phi_b+phi_b_n) + phi_0

    return loss



################ Get gradient of the loss function ################
def get_grad(model, x_r, x_b, x_0, X_b_n, current_values,discount_factor):
    #discount_factor=1.0
    current_values=-1.0
    with tf.GradientTape(persistent=True) as tape:
        # No need to watch trainable variables manually
        # Compute the loss (assuming you have a function `compute_loss` defined)
        loss = compute_loss(model, x_r, x_b, x_0, X_b_n, current_values,discount_factor)

    # Compute the gradients of the loss with respect to the model's trainable variables
    gradients = tape.gradient(loss, model.trainable_variables)
    
    return loss, gradients

# %%
type(model.trainable_variables[-1])

# %%
import tensorflow as tf

# Check if TensorFlow can detect the GPU
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)

# %%
import tensorflow as tf
import tensorflow_probability as tfp  # Import TensorFlow Probability for L-BFGS
import time
import matplotlib.pyplot as plt

# Set checkpoint file path and device
#checkpoint_filepath = CHECKPOINT_PATH
device = '/GPU:0'  # Specify the device to run on (e.g., GPU)

# Initialize model, optimizer, and training setup within the device context
with tf.device(device):
    # Initialize model
    model = init_model()

    # Choose the optimizer (Adam for initial training phase)
    adam_optim = tf.keras.optimizers.Adam()

    # Define one training step for Adam
    @tf.function  # Defining the training step as a TensorFlow function
    def train_step_adam(optimizer):
        # Compute current loss and gradient w.r.t. parameters
        loss, grad_theta = get_grad(model, X_r, X_b, X_0, X_b_n, current_values,discount_factor)

        # Perform gradient descent step
        optimizer.apply_gradients(zip(grad_theta, model.trainable_variables))

        return loss

    # Define one training step for L-BFGS (uses tensorflow-probability optimizer)
    def train_step_lbfgs():
        def loss_and_grads(position):
            # Set model parameters to the current position
            start = 0
            for var in model.trainable_variables:
                var_shape = tf.shape(var)
                size = tf.reduce_prod(var_shape)
                new_value = tf.reshape(position[start:start + size], var_shape)
                var.assign(new_value)
                start += size

            # Compute loss
            with tf.GradientTape() as tape:
                loss, _ = get_grad(model, X_r, X_b, X_0, X_b_n, current_values,discount_factor)

            # Compute gradients outside of the `with` block to avoid the warning
            grads = tape.gradient(loss, model.trainable_variables)
            grads_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)  # Flatten gradients

            return loss, grads_flat

        # Initial position (flatten model parameters)
        initial_position = tf.concat([tf.reshape(v, [-1]) for v in model.trainable_variables], axis=0)

        # Run L-BFGS optimization using TensorFlow Probability
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=loss_and_grads,
            initial_position=initial_position,
            tolerance=1e-8,  # You can adjust tolerance for convergence
            max_iterations=10  # You can adjust the number of L-BFGS iterations
        )

        # After optimization, update model variables with the L-BFGS result
        pos = results.position
        start = 0
        for var in model.trainable_variables:
            var_shape = tf.shape(var)
            size = tf.reduce_prod(var_shape)
            new_value = tf.reshape(pos[start:start + size], var_shape)
            var.assign(new_value)
            start += size

        return results.objective_value


    # Number of total training epochs
    N = NUMBER_TRAINING_EPOCHS

    # Loss history
    hist = []

    # Adam training duration (epochs)
    adam_epochs = 155000  # Define the number of epochs to run Adam optimizer
    min_loss = float('inf')  # Initialize with infinity
    patience = 10000  # Patience for early stopping during Adam phase
    wait = 1000  # Counter for epochs without improvement
    tolerance = 1e-4  # Tolerance for considering the loss as "improved"

    # Start timer
    t0 = time.time()

    # Print model summary
    model.summary()

    # Training loop
    for i in range(N + 1):
        # Phase 1: Adam optimization
        if i < adam_epochs:
            loss = train_step_adam(adam_optim)
        else:
            # Phase 2: Switch to L-BFGS optimization
            if i == adam_epochs:
                print(f"Switching to L-BFGS optimizer at epoch {i}")
            loss = train_step_lbfgs()

        # Append current loss to history
        hist.append(loss.numpy())

        # Check if the current loss is a new minimum
        #if loss.numpy() < min_loss - tolerance:
            #min_loss = loss.numpy()  # Update the minimum loss
            #wait = 0  # Reset the patience counter
        #else:
            #wait += 1  # Increment the patience counter

        # Optional: Print loss at intervals
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss.numpy()}")

        # Early stopping if no improvement
        if wait > patience:
            print("Early stopping due to lack of improvement.")
            break

    # End of training
    print(f"Training completed in {time.time() - t0:.2f} seconds")

# Plot the loss history
plt.plot(hist, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.legend()
plt.show()

# %%
import tensorflow as tf

# Define the fixed value for the first column (e.g., x = 5)
x_value = 1.0

# Create an array of shape (100, 2)
fixed_x_column = tf.fill([1200, 1], x_value)
linear_space_column = tf.linspace(0.0, 7, 1200)


# Concatenate to form the final array
final_array = tf.concat([tf.reshape(linear_space_column, [-1, 1]),fixed_x_column], axis=1)

# View the result
final_array_np = final_array.numpy()
print(final_array_np)

# %%
y_pred=model.predict(final_array)
y_pred

# %%
plt.plot(final_array_np[:,0], y_pred*30000, label='Predicted')

# %%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'  # Update this path based on your OS

# Verify that the path exists
if not os.path.exists(font_path):
    raise FileNotFoundError(f"Font file not found at {font_path}")

# Create a FontProperties object
font_prop = fm.FontProperties(fname=font_path)

# %%
np.save('neg.npy',y_pred)

# %%
# OCP and Normalized Concentrations
OCP_n = np.array([1, 0.8, 0.6, 0.496555, 0.42018, 0.365283, 0.315169, 0.257101, 0.21417, 0.211067, 
                  0.196811, 0.173801, 0.150797, 0.133356, 0.128689, 0.126398, 0.125698, 0.124966, 
                  0.123459, 0.122745, 0.118871, 0.111748, 0.091921, 0.0888005, 0.0888914, 0.0890209, 
                  0.0891475, 0.0892607, 0.0885856, 0.0895183, 0.088043, 0.0857123])
Norm_concentration_n = np.array([0.001, 0.0196774193548387, 0.0383548387096774, 0.0570322580645161, 
                                 0.0757096774193548, 0.0943870967741936, 0.113064516129032, 0.131741935483871, 
                                 0.15041935483871, 0.169096774193548, 0.187774193548387, 0.206451612903226, 
                                 0.225129032258065, 0.243806451612903, 0.262483870967742, 0.281161290322581, 
                                 0.299838709677419, 0.318516129032258, 0.337193548387097, 0.355870967741936, 
                                 0.374548387096774, 0.393225806451613, 0.411903225806452, 0.43058064516129, 
                                 0.449258064516129, 0.467935483870968, 0.486612903225806, 0.505290322580645, 
                                 0.523967741935484, 0.542645161290323, 0.561322580645161, 0.58])

OCP_p = np.array([4.19338, 4.1609, 4.12453, 4.08816, 4.0479, 4.00113, 3.96087, 3.9258, 3.88944, 
                  3.84918, 3.81414, 3.7791, 3.75449, 3.73636, 3.70913])
Norm_concentration_p = np.array([0.38, 0.413571428571429, 0.447142857142857, 0.480714285714286, 
                                 0.514285714285714, 0.547857142857143, 0.581428571428571, 0.615, 
                                 0.648571428571429, 0.682142857142857, 0.715714285714286, 0.749285714285714, 
                                 0.782857142857143, 0.816428571428571, 0.85])

# %%
import numpy as np
from scipy.interpolate import interp1d
method = 'linear'
f_n=interp1d(OCP_n,Norm_concentration_n,kind=method,fill_value='extrapolate')

# %%

# OCP and Normalized Concentrations
OCP_n = np.array([1, 0.8, 0.6, 0.496555, 0.42018, 0.365283, 0.315169, 0.257101, 0.21417, 0.211067, 
                  0.196811, 0.173801, 0.150797, 0.133356, 0.128689, 0.126398, 0.125698, 0.124966, 
                  0.123459, 0.122745, 0.118871, 0.111748, 0.091921, 0.0888005, 0.0888914, 0.0890209, 
                  0.0891475, 0.0892607, 0.0885856, 0.0895183, 0.088043, 0.0857123])
Norm_concentration_n = np.array([0.001, 0.0196774193548387, 0.0383548387096774, 0.0570322580645161, 
                                 0.0757096774193548, 0.0943870967741936, 0.113064516129032, 0.131741935483871, 
                                 0.15041935483871, 0.169096774193548, 0.187774193548387, 0.206451612903226, 
                                 0.225129032258065, 0.243806451612903, 0.262483870967742, 0.281161290322581, 
                                 0.299838709677419, 0.318516129032258, 0.337193548387097, 0.355870967741936, 
                                 0.374548387096774, 0.393225806451613, 0.411903225806452, 0.43058064516129, 
                                 0.449258064516129, 0.467935483870968, 0.486612903225806, 0.505290322580645, 
                                 0.523967741935484, 0.542645161290323, 0.561322580645161, 0.58])

OCP_p = np.array([4.19338, 4.1609, 4.12453, 4.08816, 4.0479, 4.00113, 3.96087, 3.9258, 3.88944, 
                  3.84918, 3.81414, 3.7791, 3.75449, 3.73636, 3.70913])
Norm_concentration_p = np.array([0.38, 0.413571428571429, 0.447142857142857, 0.480714285714286, 
                                 0.514285714285714, 0.547857142857143, 0.581428571428571, 0.615, 
                                 0.648571428571429, 0.682142857142857, 0.715714285714286, 0.749285714285714, 
                                 0.782857142857143, 0.816428571428571, 0.85])

# Parameters and Constants
T = 298.15  # Temperature in Kelvin
k0_p = 2e-7  # Reaction rate in pos. electrode
k0_n = 2e-6  # Reaction rate in neg. electrode

# Geometric Parameters
L_n = 1e-4  # Thickness of negative electrode (m)
L_p = 12e-5  # Thickness of positive electrode (m)
R_p = 8.28e-6  # Radius of particles in positive electrode (m)
R_n = 8.72e-6  # Radius of particles in negative electrode (m)
epsilon_n = 0.8  # Volume fraction in solid for neg. electrode
epsilon_p = 0.68  # Volume fraction in solid for pos. electrode
as_p = (3 * epsilon_p) / R_p  # Specific interfacial surface area for pos. electrode (m^2/m^3)
as_n = (3 * epsilon_n) / R_n  # Specific interfacial surface area for neg. electrode (m^2/m^3)

# Kinetic Parameters
alpha_p = 0.5  # Charge transfer coefficient for positive electrode
alpha_n = 0.5  # Charge transfer coefficient for negative electrode
Rbar = 8.314472  # Universal gas constant (J/(mol·K))
Rf_n = 5e-3  # Resistance in negative electrode (Ohms)

# Transport Parameters
Ds_n = 0.8e-14  # Diffusion coefficient in solid for neg. electrode (m^2/s)
Ds_p = 2.00E-14  # Diffusion coefficient in solid for pos. electrode (m^2/s)
F = 96487  # Faraday's constant (C/mol)
A_n = 0.07  # Surface area of negative electrode (m^2)
A_p = 0.05  # Surface area of positive electrode (m^2)

# Concentrations
Csmax_p = 5E+04  # Max concentration in cathode (mol/m^3)
Csmax_n = 3E+04  # Max concentration in anode (mol/m^3)
n_li = 0.2  # Moles of Lithium
ce = 1e3  # Fixed electrolyte concentration for SPM (mol/m^3)

# %%
1200/Csmax_n

# %%
k1 = -(epsilon_n*L_n*A_n)/(epsilon_p*L_p*A_p)
k2 = n_li/(epsilon_p*L_p*A_p)

# %%
y_pred=y_pred*Csmax_n

y_pred

# %%
y_pred_pos=((y_pred*k1)+k2)

# %%
# Create an array of different x values
x_values = [1.0, 0.95, 0.9, 0.85, 0.8,0.75]  # List of x values to plot

# Initialize plot
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 6
})
plt.figure(figsize=(3.5, 1.5))

# Store the first time-point prediction for normalization
initial_pred_value = 0.04

# Loop over different x values and plot for each
for x_value in x_values:
    # Create the fixed column for each x value
    fixed_x_column = tf.fill([1200, 1], x_value)
    linear_space_column = tf.linspace(0.0, 4.55, 1200)

    # Concatenate to form the final array
    final_array = tf.concat([tf.reshape(linear_space_column, [-1, 1]), fixed_x_column], axis=1)
    final_array_np = final_array.numpy()

    # Scale the x values from 0-5 to 0-9000 seconds
    scaled_x_values = final_array_np[:, 0] * 1800

    # Generate predicted y values using the model
    y_pred = model.predict(final_array_np)  # Replace with actual prediction logic

    # Normalize the prediction so that the first value (at t=0) is the same for all x values
    # Store the first time point prediction
    y_pred[0]=initial_pred_value 
    print(y_pred) # Scale all predictions so they start at the same initial value

    # Plot the predicted concentration of anode over time for each x_value
    plt.plot(scaled_x_values, (y_pred * 30000*k1)+k2, label=f'r = {x_value}')

# Add labels and title to the plot
plt.xlabel('Time (seconds)',fontproperties=font_prop)  # X-axis represents time in seconds
plt.ylabel('Concentration of Cathode [mol/m^3]',fontproperties=font_prop)  # Y-axis represents concentration
#plt.title('Predicted Cathode Concentration Over Time for Different Radius')  # Title of the plot

# Set the x-axis limits to range from 0 to 9000 seconds
plt.xlim([0, 9000])
plt.grid(True)

# Optionally add a legend
plt.legend(prop=font_prop)  # Font size for legend
plt.xticks(fontproperties=font_prop)  # X-axis values
plt.yticks(fontproperties=font_prop) 
plt.savefig('cathode_concetration_for different x_value.eps', format='eps', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# %%
import numpy as np
from scipy import interpolate
#ppp = interpolate(Norm_concentration_p, OCP_p)
f_p=interpolate.interp1d(Norm_concentration_p, OCP_p,kind=method,fill_value='extrapolate')
f_n=interpolate.interp1d(Norm_concentration_n, OCP_n,kind=method,fill_value='extrapolate')

# %%
ocp_n=f_n(y_pred/Csmax_n)
ocp_p=f_p(y_pred_pos/Csmax_p)

# %%
# i0_n(i) = k0_n*sqrt(ce*Cs_anode(end,i)*(Csmax_n-Cs_anode(end,i)));
    #i0_p(i) = k0_p*sqrt(ce*Cs_cathode_surface(i)*(Csmax_p-Cs_cathode_surface(i)));
i0_n = k0_n*np.sqrt(ce*y_pred*(abs(Csmax_n-y_pred)))
i0_p = k0_p*np.sqrt(ce*y_pred_pos*(abs(Csmax_p-y_pred_pos)))

# %%
   #i0_p(i) = k0_p*sqrt(ce*Cs_cathode_surface(i)*(Csmax_p-Cs_cathode_surface(i)));
i0_n = k0_n*np.sqrt(ce*y_pred*(abs(Csmax_n-y_pred)))
i0_p = k0_p*np.sqrt(ce*y_pred_pos*(abs(Csmax_p-y_pred_pos)))

# %%
tf.greater_equal(tf.range(10),tf.cast(5,tf.int32))

# %%
N_b

# %%
current_values=-4*discount_factor
discount_factor

# %%
current_values_expand=tf.expand_dims(current_values, axis=1)

# %%
current_values_expand.shape

# %%
# Term1(i) = ((Rbar*T)/(alpha_n*F))*asinh(Experimental_Current_vector(i)/(i0_n(i)*(2*as_n*A_n*L_n)));
    #Term2(i) = ((Rbar*T)/(alpha_p*F))*asinh((-Experimental_Current_vector(i))/(i0_p(i)*(2*as_p*A_p*L_p)));
    
    #Voltage_MATLAB(i) = OCP_cathode(i) - OCP_anode(i) + Term2(i) - Term1(i) - Rf_n*Experimental_Current_vector(i);
Term1 = ((Rbar*T)/(alpha_n*F))*np.arcsinh(current_values_expand/(i0_n*(2*as_n*A_n*L_n)))
Term2 = ((Rbar*T)/(alpha_p*F))*np.arcsinh(current_values_expand/(i0_p*(2*as_p*A_p*L_p)))
voltage=ocp_p-ocp_n+Term2-Term1-Rf_n*current_values_expand

# %%
voltage=voltage+0.79

# %%
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 6
})
plt.figure(figsize=(3.5, 1.5))
# Generate the time points
time_points = tf.linspace(0.0, 0.450, N_b)
plt.figure(figsize=(3.5, 1.5))
plt.plot(time_points*18000, voltage, label='Predicted')
plt.plot(time_points*18000, voltage_true_1, label='True')
plt.xlabel('Time (s)',fontproperties=font_prop)
plt.ylabel('Voltage (V)',fontproperties=font_prop)
#plt.title('Voltage vs Time (V)',fontproperties=font_prop)
plt.grid(True)
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.legend(prop=font_prop)

plt.savefig('voltage_time_overlapping.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()

# %%
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 6
})
plt.figure(figsize=(3.5, 1.5))

# Scale the x values from 0-5 to 0-9000 seconds
scaled_x_values = final_array_np[:, 0] * 1800

# Plot the predicted concentration of anode over time
plt.plot(scaled_x_values, voltage, label='Predicted')

# Add labels and title to the plot
plt.xlabel('Time (seconds)')  # X-axis represents time in seconds
plt.ylabel('Terminal Voltage')  # Y-axis represents concentration
plt.title('Predicted Terminal Voltage Over Time')  # Title of the plot

# Set the x-axis limits to range from 0 to 9000 seconds
plt.xlim([0, 9000])
plt.grid(True)

plt.xticks()  # Font size for x-ticks
plt.yticks()  # Font size for y-ticks
# Optionally add a legend
plt.legend()
plt.savefig('terminal_voltage.eps', format='eps', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()

# %%
#
#plot them

# %%
18770/Csmax_n

# %%


class KalmanFilter:
    def __init__(self, n_params, Q=1e-5, R=0.01, initial_weights=None):
        # Initialize with the actual weights of the last layer (if available)
        if initial_weights is not None:
            self.theta = tf.Variable(initial_weights)  # Use the actual weights of the last layer
        else:
            self.theta = tf.Variable(tf.zeros(n_params))  # Default initialization (zero vector)
        
        self.P = tf.eye(n_params)  # Covariance matrix (initial uncertainty)
        self.Q = tf.eye(n_params) * Q  # Process noise covariance (uncertainty in our model)
        self.R = R  # Measurement noise covariance (uncertainty in the measurements)

    def update(self, phi, y_true):
        # Kalman Filter Update Step
        y_pred = tf.reduce_sum(self.theta * phi)  # Predicted output
        
        e = y_true - y_pred  # Error (residual) between predicted and true values

        # Measurement matrix H (mapping the weights to the predicted output)
        H = tf.expand_dims(phi, 0)  # Shape: [1, n_params] (phi contains the feature vector)

        # Prediction step
        S = tf.matmul(tf.matmul(H, self.P), H, transpose_b=True) + self.R  # Innovation covariance
        K = tf.matmul(tf.matmul(self.P, H, transpose_b=True), tf.linalg.inv(S))  # Kalman gain

        # Update parameters (weights) using Kalman gain and the error
        self.theta.assign_add(K * e)
        
        # Update covariance matrix (reduce uncertainty)
        self.P.assign(tf.matmul(tf.eye(len(self.theta)) - tf.matmul(K, H), self.P))

        return y_pred, e

# %%
def calculate_voltage(y_pred, Norm_concentration_p, OCP_p, Norm_concentration_n, OCP_n, Csmax_n, Csmax_p, k0_n, k0_p, ce, Rbar, T, alpha_n, alpha_p, F, as_n, A_n, L_n, as_p, A_p, L_p, Rf_n, current_values_expand,discount_factor):
    """
    
    
    Args:
        y_pred: Predicted concentration (Tensor or numpy array)
        Norm_concentration_p: Normalized concentration for positive electrode
        OCP_p: Open Circuit Potential for positive electrode
        Norm_concentration_n: Normalized concentration for negative electrode
        OCP_n: Open Circuit Potential for negative electrode
        Csmax_n: Maximum concentration at the negative electrode
        Csmax_p: Maximum concentration at the positive electrode
        k0_n: Pre-exponential factor for negative electrode
        k0_p: Pre-exponential factor for positive electrode
        ce: Concentration of lithium in the electrolyte
        Rbar: Universal gas constant
        T: Temperature
        alpha_n: Charge transfer coefficient for the negative electrode
        alpha_p: Charge transfer coefficient for the positive electrode
        F: Faraday constant
        as_n: Surface area of negative electrode
        A_n: Area of negative electrode
        L_n: Length of negative electrode
        as_p: Surface area of positive electrode
        A_p: Area of positive electrode
        L_p: Length of positive electrode
        Rf_n: Resistance of the negative electrode
        discount_factor: Discount factor for current
    
    Returns:
        voltage: Calculated voltage
    """
    
    # First, calculate the predicted concentration and associated voltage terms
   # y_pred_neg = y_pred * 300000
    k1 = -(epsilon_n * L_n * A_n) / (epsilon_p * L_p * A_p)
    k2 = n_li / (epsilon_p * L_p * A_p)
    y_pred_pos = ((y_pred * k1) + k2)

    # Interpolate to get the OCP values
    f_p = interpolate.interp1d(Norm_concentration_p, OCP_p, kind='linear', fill_value='extrapolate')
    f_n = interpolate.interp1d(Norm_concentration_n, OCP_n, kind='linear', fill_value='extrapolate')

    ocp_n = f_n(y_pred / Csmax_n)
    ocp_p = f_p(y_pred_pos / Csmax_p)

    # Calculate i0_n and i0_p
    i0_n = k0_n * np.sqrt(ce * y_pred * (abs(Csmax_n - y_pred)))
    i0_p = k0_p * np.sqrt(ce * y_pred_pos * (abs(Csmax_p - y_pred_pos)))

    # Compute the current values using the discount factor
    current_values_expand=discount_factor*current_values_expand

    # Calculate Term1 and Term2 for voltage calculation
    Term1 = ((Rbar * T) / (alpha_n * F)) * np.arcsinh(current_values_expand / (i0_n * (2 * as_n * A_n * L_n)))
    Term2 = ((Rbar * T) / (alpha_p * F)) * np.arcsinh(current_values_expand / (i0_p * (2 * as_p * A_p * L_p)))

    # Calculate the voltage
    voltage = ocp_p - ocp_n + Term2 - Term1 - Rf_n * current_values_expand

    return voltage

# %%
# Kalman Filter class to adapt the final layer using real-time error
class KalmanFilter:
    def __init__(self, n_params, Q=1e-5, R=0.01, initial_weights=None):
        # Initialize with the actual weights of the last layer (if available)
        if initial_weights is not None:
            self.theta = tf.Variable(initial_weights)  # Use the actual weights of the last layer
        else:
            self.theta = tf.Variable(tf.zeros(n_params))  # Default initialization (zero vector)
        
        self.P = tf.eye(n_params)  # Covariance matrix (initial uncertainty)
        self.Q = tf.eye(n_params) * Q  # Process noise covariance (uncertainty in our model)
        self.R = R  # Measurement noise covariance (uncertainty in the measurements)

    def update(self, phi, voltage_true,current_values_expand,discount_factor):
        # Kalman Filter Update Step
        y_pred = tf.reduce_sum(self.theta * phi)  # Predicted output
        voltage_pred=calculate_voltage(y_pred*30000, Norm_concentration_p, OCP_p, Norm_concentration_n, OCP_n, Csmax_n, Csmax_p, k0_n, k0_p, ce, Rbar, T, alpha_n, alpha_p, F, as_n, A_n, L_n, as_p, A_p, L_p, Rf_n,current_values_expand, discount_factor)
        
        e = voltage_true - voltage_pred # Error (residual) between predicted and true values

        # Measurement matrix H (mapping the weights to the predicted output)
        H = tf.expand_dims(phi, 0)  # Shape: [1, n_params] (phi contains the feature vector)

        # Prediction step
        S = tf.matmul(tf.matmul(H, self.P), H, transpose_b=True) + self.R  # Innovation covariance
        K = tf.matmul(tf.matmul(self.P, H, transpose_b=True), tf.linalg.inv(S))  # Kalman gain

        # Update parameters (weights) using Kalman gain and the error
        self.theta.assign_add(K * e)
        
        # Update covariance matrix (reduce uncertainty)
        self.P.assign(tf.matmul(tf.eye(len(self.theta)) - tf.matmul(K, H), self.P))

        return y_pred, voltage_pred,e

# %%
final_array_np

# %%
initial_weights = model.layers[-1].get_weights()[0]  # Get the weights of the last layer
kalman_filter = KalmanFilter(n_params=initial_weights.shape[0], Q=1e-5, R=0.01, initial_weights=initial_weights)
new_current_values = current_values_expand * 0.5

# %%
calculate_voltage(y_pred*30000, Norm_concentration_p, OCP_p, Norm_concentration_n, OCP_n, Csmax_n, Csmax_p, k0_n, k0_p, ce, Rbar, T, alpha_n, alpha_p, F, as_n, A_n, L_n, as_p, A_p, L_p, Rf_n, current_values_expand,discount_factor_for_current)

# %%
dataset = tf.data.TextLineDataset("SPM_Results_Scale_1.00.txt")
true_voltage=[]
for line_tensor in dataset.skip(1):  # Skip the header line
    # Decode the tensor to a string
    line = line_tensor.numpy().decode('utf-8')
    time_voltage = line.split()
    try:
        voltage = float(time_voltage[1])
        true_voltage.append(voltage)
    except ValueError:
        continue  # Skip lines that cannot be converted to float

# %%
true_voltage=true_voltage[:-1]
new_indices=np.linspace(0, len(true_voltage)-1, len(final_array_np[:,0]))
interpolated_true_voltage = interp1d(np.arange(len(true_voltage)), true_voltage, kind='linear')
voltage_true_1=interpolated_true_voltage(new_indices)

# %%
len(voltage_true_1)

# %%
model.layers[-1].get_weights()

# %%
model.summary()

# %%
class KalmanFilter:
    def __init__(self, n_params, Q=1e-4, R=0.001, initial_weights=None):
        # Initialize weights
        if initial_weights is not None:
            self.theta = tf.Variable(initial_weights)  # Initial weights
        else:
            self.theta = tf.Variable(tf.zeros(n_params + 1))  # Include bias term

        # Adjust dimensions of P and Q to include bias term
        self.P = tf.Variable(tf.eye(n_params)*0.01, trainable=False)  # Covariance matrix
        self.Q = tf.Variable(tf.eye(n_params) * Q, trainable=False)  # Process noise covariance
        self.R = tf.reshape(tf.constant(R, dtype=tf.float32), [1, 1])  # Measurement noise covariance

    def update(self, phi, voltage_true, current_values_expand, discount_factor):
        # Predicted concentration
        y_pred = tf.matmul(phi, self.theta)  # Predicted output

        # Voltage prediction using the physical model
        voltage_pred = calculate_voltage(
            y_pred * 30000, Norm_concentration_p, OCP_p, Norm_concentration_n, OCP_n,
            Csmax_n, Csmax_p, k0_n, k0_p, ce, Rbar, T, alpha_n, alpha_p, F,
            as_n, A_n, L_n, as_p, A_p, L_p, Rf_n, current_values_expand, discount_factor
        )
        
        # Compute residual (error)
        e = tf.reshape(voltage_true - voltage_pred, [1, 1])  # Ensure shape [1, 1]

        # Measurement matrix H
        H = tf.expand_dims(phi, axis=0)  # Shape: [1, n_params + 1]

        # Prediction step
        self.P.assign(self.P + 1e-6 * tf.eye(self.P.shape[0]))  # Small regularization term
        S = tf.matmul(tf.matmul(H, self.P), H, transpose_b=True) + self.R  # Innovation covariance
        K = tf.matmul(tf.matmul(self.P, H, transpose_b=True), tf.linalg.inv(S))  # Kalman gain
        
        # Clamp the Kalman gain to prevent instability
        K = tf.clip_by_value(K, clip_value_min=-0.1, clip_value_max=0.1)  # Adjust min/max based on the system

        # Update weights using Kalman gain and residual
        delta = tf.reshape(K * e, self.theta.shape)  # Ensure shape matches theta
        self.theta.assign_add(delta)  # Update weights

        # Update covariance matrix
        I = tf.eye(self.P.shape[0])  # Identity matrix
        updated_P = tf.matmul(I - tf.matmul(K, H), self.P)  # Compute updated covariance
        self.P.assign(tf.squeeze(updated_P))  # Remove extra dimension if it exists
        
        # Print debug info for tracking progress
        print(self.P)
        print(self.theta)
        print(y_pred)

        return y_pred, voltage_pred, e

# %%
initial_weights = tf.concat([ model.layers[-1].get_weights()[0],model.layers[-1].get_weights()[1].reshape(-1, 1)  ],axis=0) # Get the weights of the last layer
kalman_filter = KalmanFilter(n_params=initial_weights.shape[0], Q=1e-4, R=0.01, initial_weights=initial_weights)

# %%
initial_weights.shape[0]

# %%
voltage_true_1.shape

# %%
discount_factor_for_current=1.25

# %%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'  # Update this path based on your OS

# Verify that the path exists
if not os.path.exists(font_path):
    raise FileNotFoundError(f"Font file not found at {font_path}")

# Create a FontProperties object
font_prop = fm.FontProperties(fname=font_path)

# %%
voltage_pred_kalmas_all=np.zeros(len(final_array_np[:,0]))
erros=np.zeros(len(final_array_np[:,0]))


for i in range(len(final_array_np[:,0])):
    x_input = final_array_np[i] # Example input features
    x_input_tensor = tf.convert_to_tensor(x_input.reshape(1,-1), dtype=tf.float32)
    z1 = model.layers[1](x_input_tensor)  # Output of dense_3
    z2 = model.layers[2](z1)  # Output of dense_4
    z3 = model.layers[3](z2)  # Output of dense_5 (final output layer)
    ones_tensor = tf.ones((tf.shape(z2)[0], 1), dtype=tf.float32)
    phi = tf.concat([z2, ones_tensor], axis=1)
    theta = model.layers[-1].get_weights()[0]
    bias = model.layers[-1].get_weights()[1].reshape(-1, 1)  # Reshape bias to match the rank of theta
    theta_with_bias = tf.concat([theta, bias], axis=0)
    concentraction_pred,voltage_pred_kalman,error=kalman_filter.update(phi, voltage_true_1[i],current_values_expand[i],discount_factor_for_current)
    voltage_pred_kalmas_all[i]=voltage_pred_kalman
    erros[i]=error
    if tf.abs(error) < 1e-6:  # Check if the error is sufficiently small
        break

# %%
voltage_pred_kalmas_all

# %%
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 6
})
plt.figure(figsize=(3.5, 1.5))
plt.plot(time_points*18000, voltage_true_1, label='True')
plt.plot(time_points*18000, voltage_pred_kalmas_all, label='Predicted', linestyle='--')
plt.xlabel('Time (s)', fontproperties=font_prop)
plt.ylabel('Voltage (V)', fontproperties=font_prop)
#plt.title('True Voltage vs Predicted Voltage')
plt.grid(True)
plt.xticks(fontproperties=font_prop,)
plt.yticks(fontproperties=font_prop)
plt.legend(prop=font_prop)

plt.savefig('true_voltage vs predict 1.25.eps', format='eps', dpi=300, bbox_inches='tight')

# %%
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 6
})
plt.figure(figsize=(3.5, 1.5))
plt.plot(time_points*18000, erros, label='Error')
plt.xlabel('Time (s)', fontproperties=font_prop)
plt.ylabel('Error', fontproperties=font_prop)
#plt.title('Error vs Time')
plt.grid(True)
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.savefig('error_1.25.eps', format='eps', dpi=300, bbox_inches='tight')
