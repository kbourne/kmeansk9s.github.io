### TimeGANs key functionality

## Necessary Packages
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

## Min Max Normalizer

def MinMaxScaler2(dataX):
    
    min_val = np.min(np.min(dataX, axis = 0), axis = 0)
    dataX = dataX - min_val
    
    max_val = np.max(np.max(dataX, axis = 0), axis = 0)
    dataX = dataX / (max_val + 1e-7)
    
    return dataX, min_val, max_val

## Start TimeGAN function (Input: Original data, Output: Synthetic Data)

def timegan (dataX, parameters):
  
    # Initialization on the Graph
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0,:])
    
    # Maximum seq length and each seq length
    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))
        
    # Normalization
    if ((np.max(dataX) > 1) | (np.min(dataX) < 0)):
        dataX, min_val, max_val = MinMaxScaler2(dataX)
        Normalization_Flag = 1
    else:
        Normalization_Flag = 0
     
    # Network Parameters
    hidden_dim   = parameters['hidden_dim'] 
    num_layers   = parameters['num_layers']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    module_name  = parameters['module_name']    # 'lstm' or 'lstmLN'
    z_dim        = parameters['z_dim']
    gamma        = 1
    
    ## input place holders
    tf.compat.v1.disable_eager_execution()
    X = tf.compat.v1.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, Max_Seq_Len, z_dim], name = "myinput_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name = "myinput_t")
    
    ## Basic RNN Cell
          
    def rnn_cell(module_name):
      # GRU
        if (module_name == 'gru'):
            rnn_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
      # LSTM
        elif (module_name == 'lstm'):
            rnn_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
      # LSTM Layer Normalization
        elif (module_name == 'lstmLN'):
            rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
        return rnn_cell
      
        
    ## build a RNN embedding network      
    
    def embedder (X, T):      
      
        with tf.compat.v1.variable_scope("embedder", reuse = tf.compat.v1.AUTO_REUSE):
            
            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
            
            H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)

        return H
      
    ##### Recovery
    
    def recovery (H, T):      
      
        with tf.compat.v1.variable_scope("recovery", reuse = tf.compat.v1.AUTO_REUSE):       
              
            r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            r_outputs, r_last_states = tf.compat.v1.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
            
            X_tilde = tf.contrib.layers.fully_connected(r_outputs, data_dim, activation_fn=tf.nn.sigmoid) 

        return X_tilde
    
    
    
    ## build a RNN generator network
    
    def generator (Z, T):      
      
        with tf.compat.v1.variable_scope("generator", reuse = tf.compat.v1.AUTO_REUSE):
            
            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
            
            E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     

        return E
      
    def supervisor (H, T):      
      
        with tf.compat.v1.variable_scope("supervisor", reuse = tf.compat.v1.AUTO_REUSE):
            
            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers-1)])
                
            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
            
            S = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     

        return S
      
      
      
    ## builde a RNN discriminator network 
    
    def discriminator (H, T):
      
        with tf.compat.v1.variable_scope("discriminator", reuse = tf.compat.v1.AUTO_REUSE):
            
            d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name) for _ in range(num_layers)])
                
            d_outputs, d_last_states = tf.compat.v1.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
            
            Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None) 
    
        return Y_hat   
    
    
    ## Random vector generation
    def random_generator (batch_size, z_dim, T_mb, Max_Seq_Len):
      
        Z_mb = list()
        
        for i in range(batch_size):
            
            Temp = np.zeros([Max_Seq_Len, z_dim])
            
            Temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
        
            Temp[:T_mb[i],:] = Temp_Z
            
            Z_mb.append(Temp_Z)
      
        return Z_mb
    
    ## Functions
    
    # Embedder Networks
    H = embedder(X, T)
    X_tilde = recovery(H, T)
    
    # Generator
    E_hat = generator(Z, T)
    H_hat = supervisor(E_hat, T)
    H_hat_supervise = supervisor(H, T)
    
    # Synthetic data
    X_hat = recovery(H_hat, T)
    
    # Discriminator
    Y_fake = discriminator(H_hat, T)
    Y_real = discriminator(H, T)     
    Y_fake_e = discriminator(E_hat, T)
    
    # Variables        
    e_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('recovery')]
    g_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]
    
    # Loss for the discriminator
    D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
    # Loss for the generator
    # 1. Adversarial loss
    G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    
    # 2. Supervised loss
    G_loss_S = tf.compat.v1.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,1:,:])
    
    # 3. Two Momments
    G_loss_V1 = tf.reduce_mean(input_tensor=np.abs(tf.sqrt(tf.nn.moments(x=X_hat,axes=[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(x=X,axes=[0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(input_tensor=np.abs((tf.nn.moments(x=X_hat,axes=[0])[0]) - (tf.nn.moments(x=X,axes=[0])[0])))

    G_loss_V = G_loss_V1 + G_loss_V2
    
    # Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V 
            
    # Loss for the embedder network
    E_loss_T0 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10*tf.sqrt(E_loss_T0)
    E_loss = E_loss0  + 0.1*G_loss_S
    
    # optimizer
    E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
    E_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)      
    GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)   
        
    ## Sessions    
    
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    ## Embedding Learning
    
    print('Start Embedding Network Training')
    
    for itt in range(iterations):
        
        # Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
            
        X_mb = list(dataX[i] for i in train_idx)
        T_mb = list(dataT[i] for i in train_idx)
            
        # Train embedder        
        _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})
        
        if itt % 1000 == 0:
            print('step: '+ str(itt) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )        
            
    print('Finish Embedding Network Training')
    
    ## Training Supervised Loss First
    
    print('Start Training with Supervised Loss Only')
    
    for itt in range(iterations):
        
        # Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
            
        X_mb = list(dataX[i] for i in train_idx)
        T_mb = list(dataT[i] for i in train_idx)        
        
        Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
        
        # Train generator       
        _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
                           
        if itt % 1000 == 0:
            print('step: '+ str(itt) + ', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)) )
                
    print('Finish Training with Supervised Loss Only')
    
    ## Joint Training
    
    print('Start Joint Training')
    
    # Training step
    for itt in range(iterations):
      
        # Generator Training
        for kk in range(2):
          
            # Batch setting
            idx = np.random.permutation(No)
            train_idx = idx[:batch_size]     
            
            X_mb = list(dataX[i] for i in train_idx)
            T_mb = list(dataT[i] for i in train_idx)
            
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
              
            # Train generator
            _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
            
            # Train embedder        
            _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})   
           
        ## Discriminator Training
        
        # Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
        
        X_mb = list(dataX[i] for i in train_idx)
        T_mb = list(dataT[i] for i in train_idx)
        
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, Max_Seq_Len)
            
        
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
        # Train discriminator
        
        if (check_d_loss > 0.15):        
            _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
        ## Checkpoints
        if itt % 1000 == 0:
            print('step: '+ str(itt) + 
                  ', d_loss: ' + str(np.round(step_d_loss,4)) + 
                  ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) + 
                  ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
                  ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
                  ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )
   
    
    print('Finish Joint Training')
    
    ## Final Outputs
    
    Z_mb = random_generator(No, z_dim, dataT, Max_Seq_Len)
    
    X_hat_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: dataX, T: dataT})    
    
    ## List of the final outputs
    
    dataX_hat = list()
    
    for i in range(No):
        Temp = X_hat_curr[i,:dataT[i],:]
        dataX_hat.append(Temp)
        
    # Renormalization
    if (Normalization_Flag == 1):
        dataX_hat = dataX_hat * max_val
        dataX_hat = dataX_hat + min_val
    
    return dataX_hat

## Post-hoc RNN Classifier 

def discriminative_score_metrics (dataX, dataX_hat):
  
    # Initialization on the Graph
    tf.reset_default_graph()

    # Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0,:])
    
    # Compute Maximum seq length and each seq length
    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))
     
    # Network Parameters
    hidden_dim = max(int(data_dim/2),1)
    iterations = 2000
    batch_size = 128
    
    ## input place holders
    # Features
    X = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x")
    X_hat = tf.placeholder(tf.float32, [None, Max_Seq_Len, data_dim], name = "myinput_x_hat")
    
    # Times
    T = tf.placeholder(tf.int32, [None], name = "myinput_t")
    T_hat = tf.placeholder(tf.int32, [None], name = "myinput_t_hat")
    
    ## builde a RNN classification network 
    
    def discriminator (X, T):
      
        with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE) as vs:
            
            d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'cd_cell')
                    
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, X, dtype=tf.float32, sequence_length = T)
                
            # Logits
            Y_hat = tf.contrib.layers.fully_connected(d_last_states, 1, activation_fn=None) 
            
            # Sigmoid output
            Y_hat_Final = tf.nn.sigmoid(Y_hat)
            
            # Variables
            d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    
        return Y_hat, Y_hat_Final, d_vars
    
    ## Train / Test Division
    def train_test_divide (dataX, dataX_hat, dataT):
      
        # Divide train/test index
        No = len(dataX)
        idx = np.random.permutation(No)
        train_idx = idx[:int(No*0.8)]
        test_idx = idx[int(No*0.8):]
        
        # Train and Test X
        trainX = [dataX[i] for i in train_idx]
        trainX_hat = [dataX_hat[i] for i in train_idx]
        
        testX = [dataX[i] for i in test_idx]
        testX_hat = [dataX_hat[i] for i in test_idx]
        
        # Train and Test T
        trainT = [dataT[i] for i in train_idx]
        testT = [dataT[i] for i in test_idx]
      
        return trainX, trainX_hat, testX, testX_hat, trainT, testT
    
    ## Functions
    # Variables
    Y_real, Y_pred_real, d_vars = discriminator(X, T)
    Y_fake, Y_pred_fake, _ = discriminator(X_hat, T_hat)
        
    # Loss for the discriminator
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_real, labels = tf.ones_like(Y_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_fake, labels = tf.zeros_like(Y_fake)))
    D_loss = D_loss_real + D_loss_fake
    
    # optimizer
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
        
    ## Sessions    

    # Start session and initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Train / Test Division
    trainX, trainX_hat, testX, testX_hat, trainT, testT = train_test_divide (dataX, dataX_hat, dataT)
    
    # Training step
    for itt in range(iterations):
          
        # Batch setting
        idx = np.random.permutation(len(trainX))
        train_idx = idx[:batch_size]     
            
        X_mb = list(trainX[i] for i in train_idx)
        T_mb = list(trainT[i] for i in train_idx)
        
        # Batch setting
        idx = np.random.permutation(len(trainX_hat))
        train_idx = idx[:batch_size]     
            
        X_hat_mb = list(trainX_hat[i] for i in train_idx)
        T_hat_mb = list(trainT[i] for i in train_idx)
          
        # Train discriminator
        _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})            
        
        ## Checkpoints
        if itt % 500 == 0:
           print("[step: {}] loss - d loss: {}".format(itt, np.round(step_d_loss,4)))
    
    ## Final Outputs (ontTesting set)
    
    Y_pred_real_curr, Y_pred_fake_curr = sess.run([Y_pred_real, Y_pred_fake], feed_dict={X: testX, T: testT, X_hat: testX_hat, T_hat: testT})
    
    Y_pred_final = np.squeeze(np.concatenate((Y_pred_real_curr, Y_pred_fake_curr), axis = 0))
    Y_label_final = np.concatenate((np.ones([len(Y_pred_real_curr),]), np.zeros([len(Y_pred_real_curr),])), axis = 0)
    
    ## Accuracy
    Acc = accuracy_score(Y_label_final, Y_pred_final>0.5)
    
    Disc_Score = np.abs(0.5-Acc)
    
    return Disc_Score

## Post-hoc RNN one-step ahead predictor

def predictive_score_metrics (dataX, dataX_hat):
  
    # Initialization on the Graph
    tf.reset_default_graph()

    # Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0,:])
    
    # Maximum seq length and each seq length
    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))
     
    # Network Parameters
    hidden_dim = max(int(data_dim/2),1)
    iterations = 5000
    batch_size = 128
    
    ## input place holders
    
    X = tf.placeholder(tf.float32, [None, Max_Seq_Len-1, data_dim-1], name = "myinput_x")
    T = tf.placeholder(tf.int32, [None], name = "myinput_t")    
    Y = tf.placeholder(tf.float32, [None, Max_Seq_Len-1, 1], name = "myinput_y")
    
    ## builde a RNN discriminator network 
    
    def predictor (X, T):
      
        with tf.variable_scope("predictor", reuse = tf.AUTO_REUSE) as vs:
            
            d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
                    
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, X, dtype=tf.float32, sequence_length = T)
                
            Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None) 
            
            Y_hat_Final = tf.nn.sigmoid(Y_hat)
            
            d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    
        return Y_hat_Final, d_vars
    
    ## Functions
    # Variables
    Y_pred, d_vars = predictor(X, T)
        
    # Loss for the predictor
    D_loss = tf.losses.absolute_difference(Y, Y_pred)
    
    # optimizer
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
        
    ## Sessions    

    # Session start
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Training using Synthetic dataset
    for itt in range(iterations):
          
        # Batch setting
        idx = np.random.permutation(len(dataX_hat))
        train_idx = idx[:batch_size]     
            
        X_mb = list(dataX_hat[i][:-1,:(data_dim-1)] for i in train_idx)
        T_mb = list(dataT[i]-1 for i in train_idx)

        Y_mb = list(np.reshape(dataX_hat[i][1:,(data_dim-1)],[len(dataX_hat[i][1:,(data_dim-1)]),1]) for i in train_idx)        
          
        # Train discriminator
        _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})  
       
        ## Checkpoints
        if itt % 500 == 0:
           print("[step: {}] loss - d loss: {}".format(itt, np.sqrt(np.round(step_d_loss,4))))
    
    ## Use Original Dataset to test
    
    # Make Batch with Original Data
    idx = np.random.permutation(len(dataX_hat))
    train_idx = idx[:No]     
    
    X_mb = list(dataX[i][:-1,:(data_dim-1)] for i in train_idx)
    T_mb = list(dataT[i]-1 for i in train_idx)
    Y_mb = list(np.reshape(dataX[i][1:,(data_dim-1)], [len(dataX[i][1:,(data_dim-1)]),1]) for i in train_idx)
    
    # Predict Fugure
    pred_Y_curr = sess.run(Y_pred, feed_dict={X: X_mb, T: T_mb})
    
    # Compute MAE
    MAE_Temp = 0
    for i in range(No):
        MAE_Temp = MAE_Temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
    
    MAE = MAE_Temp / No
    
    return MAE

## Min Max Normalizer

def MinMaxScaler3(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def doggo_data_timeGANs(df, seq_length=1):

    # remove non-feature columns
    drop_cols = ['dog_id', 'hs_health_conditions_cancer', 'hs_condition_type', 'hs_diagnosis_month', 'hs_diagnosis_year']
    df = df.drop(drop_cols, axis=1)
    print('doggo_data_timeGANs dataframe shape post: ', df.shape)
    
    # turn into segment vector
    x = df.to_numpy()
    
    # Flip the data to make chronological data
    x = x[::-1]
    
    # Min-Max Normalizer
    x = MinMaxScaler3(x)
    
    # Build dataset
    dataX = []
    
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to random sample)
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    
    return outputX

def get_id(df):
    idclass = NumAdv()
    myiter = iter(idclass)
    dog_id = ((next(myiter)) + 100000) # original dog ids max at 92582
    return dog_id

def prep_synth_data(dataX_hat):

    list_of_lists = []

    for arr in dataX_hat: #dataX
        for outer_lst in arr:
            list_of_lists.append(outer_lst)

    syn_df = pd.DataFrame(list_of_lists)

    # add a consecutive id starting at 100,000 for every 4 rows
    syn_df['dog_id'] = (syn_df.index / 4 + 100000).astype(int)

    return syn_df

# take original TimeGANs loaded data as usual, 
#   but then filter by 'hs_health_conditions_cancer'==1
#   before passing into TimeGANs 
def oversamp_minority_data(gan_df, seqlen=4):

    gan_df_min = gan_df[gan_df['hs_health_conditions_cancer'] == 1]

    # so run original timeGANS data changer, but on smaller df
    dataX_min = doggo_data_timeGANs(gan_df_min, seq_length=seqlen)

    print('minority doggo dataset is ready')
    
    return dataX_min

class NumAdv:
    def __iter__(self):
        self.a = int(1)
        return self
    
    def __next__(self):
        x = self.a
        self.a += 1
        if self.a % 2==0:
            fin = (x / 2)
        else:
            fin = ((x - 1) / 2)
        return fin * 2