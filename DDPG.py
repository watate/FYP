import tensorflow as tf
import numpy as np
import rospy
import random
import time
import copy
import pickle
import os

from StageWorld import StageWorld
from ReplayBuffer import ReplayBuffer
from noise import Noise
from reward import Reward
from actor import ActorNetwork
from critic import CriticNetwork

import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ==========================
#   Training Parameters
# ==========================
# Maximum episodes run
MAX_EPISODES = 50000
# Episodes with noise
#NOISE_MAX_EP = 1000
NOISE_MAX_EP = 1
# Noise parameters - Ornstein Uhlenbeck
DELTA = 0.5 # The rate of change (time)
SIGMA = 0.5 # Volatility of the stochastic processes
OU_A = 3. # The rate of mean reversion
OU_MU = 0. # The long run average interest rate

# E-gready
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.001 # starting value of epsilon

# Reward parameters
REWARD_FACTOR = 0.1 # Total episode reward factor
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

LASER_BEAM = 40
#LASER_HIST is laser history. Makes DDPG take into account current laser and previous laser scans.
#LASER_HIST = 3 means that DDPG has to account for 3 scans (current, previous, previous previous)
LASER_HIST = 3
ACTION = 2
TARGET = 2
SPEED = 2
SWITCH = 2
ACCEL = 2
JERK = 2

SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
BUFFER_SIZE = 20000
VAJ_BUFFER_SIZE = 30
MINIBATCH_SIZE = 32

GAME = 'StageWorld'

#map_name = 'Obstacles'
map_name = 'Obstacles3'
#map_name = 'Blank'
map_type = './worlds/' + map_name + '.jpg'

save_frequency = 500
prevent_PID = 1

save_velocity_bool = 1 #don't forget to change save_frequency to save velocity
load_replay_buffer_bool = 0
velocity_list_filename = 'jerk_complex_velocity_list.dat'
replay_buffer_filename = 'replay_buffer.dat'
current_date_time = time.strftime("%Y%m%d-%H%M%S")

config_notes = """Collecting velocity data for Jerk Complex Model""" #Write stuff here to explain what training was for
# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.compat.v1.Summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.compat.v1.Summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.compat.v1.Summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, noise, reward, discrete, action_bound):
    # Set up summary writer
    summary_writer = tf.compat.v1.summary.FileWriter("ddpg_summary/" + current_date_time)
    #add sessions graph to our summary writer
    summary_writer.add_graph(sess.graph)

    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    print('checkpoint:', checkpoint)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Check if replay buffer exists, reload it if it does
    if load_replay_buffer_bool == 1:
        if os.path.exists(replay_buffer_filename):
            with open(replay_buffer_filename, 'rb') as rfp:
                buff = pickle.load(rfp)

    if save_velocity_bool == 1:        
        velocity_list = list()

    # first time you run this, "velocity_list.dat" won't exist
    # so we need to check for its existence before we load 
    # our "database"
    if save_velocity_bool == 1:
        if os.path.exists(velocity_list_filename):
            with open(velocity_list_filename, 'rb') as rfp:
                velocity_list = pickle.load(rfp)



    # plot settings
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(env.map, aspect='auto', cmap='hot', vmin=0., vmax=1.5)
    plt.show(block=False)

    # Initialize noise
    ou_level = 0.
    epsilon = INITIAL_EPSILON

    rate = rospy.Rate(5) #forces code to loop exactly 5 times per second
    loop_time = time.time() #returns time in seconds
    last_loop_time = loop_time
    i = 0 #initialize i to count episodes

    s1 = env.GetLaserObservation()
    s_1 = np.stack((s1, s1, s1), axis=1)

    T = 0
    for i in range(MAX_EPISODES):
        env.ResetWorld()
        env.GenerateTargetPoint()
        print 'Target: (%.4f, %.4f)' % (env.target_point[0], env.target_point[1])
        target_distance = copy.deepcopy(env.distance)
        ep_reward = 0.
        ep_ave_max_q = 0. #another terribly named variable. 
                            # ep_ave_max_q is not episode average (even though the name implies it)
                            # but strictly the max (it accumulates and is later divided to find average)
                                # That is, ep_ave_max_q += max of predicted q (see below)
                            # See how Qmax is calculated in tf.summary to understand why this name doesn't make sense
                                # Qmax is ep_ave_max_q / j (where j is number of timesteps)
                                # So Qmax is the average maximum Q-value of an episode
        loop_time_buf = []
        terminal = False

        j = 0 #j is time or timestep (who labeled time as j though)
        ep_reward = 0
        ep_ave_max_q = 0
        ep_PID_count = 0.
        ep_jerk_linear = 0
        ep_jerk_angular = 0
        speed2 = (0,0)
        speed3 = (0,0)
        accel2 = (0,0)
        accel3 = (0,0)
        jerk2 = (0,0)
        jerk3 = (0,0)
        while not terminal and not rospy.is_shutdown():
            s1 = env.GetLaserObservation() #s1 is current laser observation
            s_1 = np.append(np.reshape(s1, (LASER_BEAM, 1)), s_1[:, :(LASER_HIST - 1)], axis=1)
            s__1 = np.reshape(s_1, (LASER_BEAM * LASER_HIST)) 
            target1 = env.GetLocalTarget()
            speed1 = env.GetSelfSpeed()

            ###################################################################################
            #Generate new VAJ values
            if j >= 2:
                linear_accel = speed1[0] - speed2[0]
                angular_accel = speed1[1] - speed2[1]
                linear_jerk = linear_accel - accel2[0]
                angular_jerk = angular_accel - accel2[1]
        
            else:
                linear_accel = 0
                angular_accel = 0
                linear_jerk = 0
                angular_jerk = 0
            accel1 = (linear_accel, angular_accel)
            jerk1 = (linear_jerk, angular_jerk)
            env.UpdateAccelAndJerk([linear_accel, angular_accel], [linear_jerk, angular_jerk])
        
            #Add previous history to variables
            #extend is like append but for multiple items
            if j >= 2:
                speed_stack = (speed1[0], speed1[1], speed2[0], speed2[1], speed3[0], speed3[1])
                accel_stack = (accel1[0], accel1[1], accel2[0], accel2[1], accel3[0], accel3[1]) #linear accel is item 2, angular accel is item 3
                jerk_stack = (jerk1[0], jerk1[1], jerk2[0], jerk2[1], jerk3[0], jerk3[1]) #linear jerk is item 4, angular jerk is item 5
            else:
                speed_stack = (speed1[0], speed1[1], 0, 0, 0, 0) #wt
                accel_stack = (0, 0, 0, 0, 0, 0)
                jerk_stack = (0, 0, 0, 0, 0, 0)
            
            ###################################################################################
            #state1 = np.concatenate([s__1, speed1, target1], axis=0) #add speed and target information to state
            state1 = np.concatenate([s__1, speed_stack, accel_stack, jerk_stack, target1], axis=0)

            ###################################################################################
            # Update VAJ stack
            speed3 = speed2
            speed2 = speed1
            accel3 = accel2
            accel2 = accel1
            jerk3 = jerk2
            jerk2 = jerk1
            ###################################################################################
            [x, y, theta] =  env.GetSelfStateGT()
            map_img = env.RenderMap([[0, 0], env.target_point])
            
            r, terminal, result = env.GetRewardAndTerminate(j, jerk_stack)
            ep_reward += r
            ep_jerk_linear += abs(linear_jerk)
            ep_jerk_angular += abs(angular_jerk)
            if j > 0 :
                buff.add(state, a[0], r, state1, terminal, switch_a_t)      #Add replay buffer
            j += 1
            state = state1

            a = actor.predict(np.reshape(state, (1, actor.s_dim)))
            switch_a = critic.predict_switch(np.reshape(state, (1, actor.s_dim))) #switch critic
            switch_a_t = np.zeros([SWITCH])
            # Add exploration noise
            if i < NOISE_MAX_EP:
                ou_level = noise.ornstein_uhlenbeck_level(ou_level)
                a = a + ou_level

                #print("a[0][1] is {}".format(a[0][1]))
                #bound action and noise to limits
                if a[0][0] > action_bound[0]:
                    a[0][0] = action_bound[0]
                if a[0][1] > action_bound[1]:
                    a[0][1] = action_bound[1]
                if a[0][1] < -action_bound[1]:
                    a[0][1] = -action_bound[1]

                #print("a[0][1] is {}".format(a[0][1]))


            if random.random() <= epsilon:
                print("----------Random Switch Action----------")
                switch_index = random.randrange(SWITCH)
                switch_a_t[switch_index] = 1
            else:
                switch_index = np.argmax(switch_a[0]) #switch_index gets the index of the largest value in switch_a[0]
                switch_a_t[switch_index] = 1 #switch_a_t

            if prevent_PID == 1:
                switch_index = 0 #prevents PID control
            if switch_index == 1:
	            a = env.PIDController(action_bound)
	            ep_PID_count += 1.
	            print("-----------PID Controller---------------")


            # Set action for discrete and continuous action spaces
            # To check action: print("Action is {}".format(a))
            # action[0] is linear x
            # action[1] is angular z
            # everything else is 0
            action = a[0]
            if  action[0] <= 0.05:
                action[0] = 0.05

            ######## VELOCITY SMOOTHER ################
            #See: https://robotics.stackexchange.com/questions/18237/what-is-a-velocity-smoother
            # target_speed = a
            # control_speed = speed1

            # print("a is {}".format(a))
            # print("action is {}".format(action))
            # print("v is {}".format(action - [0.10, 0.20]))

            # if target_speed > control_speed:
            #     control_speed = min(target_speed, control_speed + [0.10, 0.20])
            # elif target_speed < control_speed:
            #     control_speed = max(target_speed, control_speed - [0.10, 0.20])
            # else:
            #     control_speed = target_speed
            #############################################
            env.Control(action)

            ###################################################################################
            #Append to velocity list
            if save_velocity_bool == 1:
                velocity_list.append((i, action[0], action[1])) #i is to record the index (which episode is this speed from?)
            ###################################################################################

            #if save_velocity_bool == 1:
            #    velocity_list.append((action[0], action[1], ou_level)) #append velocities to list

            # plot
            if j == 1:
                im.set_array(map_img)
                fig.canvas.draw()

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if buff.count() > MINIBATCH_SIZE:
                batch = buff.getBatch(MINIBATCH_SIZE)
                s_batch = np.asarray([e[0] for e in batch])
                a_batch = np.asarray([e[1] for e in batch])
                r_batch = np.asarray([e[2] for e in batch])
                s2_batch = np.asarray([e[3] for e in batch])
                t_batch = np.asarray([e[4] for e in batch])
                switch_a_batch = np.asarray([e[5] for e in batch])
                # y_i = np.asarray([e[1] for e in batch])

                ######### Walter's Explanation #############
                # There's actually a small problem with the way discounted return is calculated
                # Here, only if buff.count > MINIBATCH_SIZE will GAMMA be used
                    # whereas in theory we should always be adding discount to each past return
                # So for k in range minibatch size
                    # if t_batch[k] is true (t_batch is an array of 'terminal', if you look at buff.add
                                            # you'll see that t_batch records whether it was terminal or not
                    # so it looks through data points in t_batch to find a terminal one
                        #and if it's terminal append r_batch (reward) to y_i
                        #otherwise, append reward + discounted q-value to y_i

                ##############################################
                # Calculate targets
                # critic
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # D3QN
                Q1 = critic.predict_switch(s2_batch)
                Q2 = critic.predict_target_switch(s2_batch)
                switch_y = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        switch_y.append(r_batch[k])
                    else:
                        switch_y.append(r_batch[k] + GAMMA * Q2[k, np.argmax(Q1[k])])
                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, switch_a_batch,\
                                                    np.reshape(y_i, (MINIBATCH_SIZE, 1)),\
                                                    switch_y)

                #np amax and max are the same, they both find the max of the array
                    #in this case, it is looking for the maximum predicted q in the episode and adding it to ep_ave_max_q
                ep_ave_max_q += np.amax(predicted_q_value)


                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs, switch_a_batch)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            last_loop_time = loop_time
            loop_time = time.time()
            loop_time_buf.append(loop_time - last_loop_time)
            T += 1
            rate.sleep()

        # scale down epsilon
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NOISE_MAX_EP

        summary = tf.compat.v1.Summary()
        summary.value.add(tag='Reward', simple_value=float(ep_reward))
        summary.value.add(tag='Qmax', simple_value=float(ep_ave_max_q / float(j)))
        summary.value.add(tag='PIDrate', simple_value=float(ep_PID_count / float(j)))
        summary.value.add(tag='Distance', simple_value=float(target_distance))
        summary.value.add(tag='Result', simple_value=float(result))
        summary.value.add(tag='Steps', simple_value=float(j))
        summary.value.add(tag='Total Linear Jerk', simple_value=float(ep_jerk_linear))
        summary.value.add(tag='Total Angular Jerk', simple_value=float(ep_jerk_angular))
        summary.value.add(tag='Episode', simple_value=float(i))

        summary_writer.add_summary(summary, T)

        summary_writer.flush()

        if i > 0 and i % save_frequency == 0 :
            print("Saving network...")
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = i)

            if save_velocity_bool == 1:
                #append to text file
                print("Saving velocities...")
                # Now we "sync" our database
                with open(velocity_list_filename,'wb') as wfp:
                    pickle.dump(velocity_list, wfp)
                # Re-load our database
                with open(velocity_list_filename,'rb') as rfp:
                    velocity_list = pickle.load(rfp)

            if load_replay_buffer_bool == 1:
                #save replay buffer
                print("Saving replay buffer...")
                # Now we "sync" our database
                with open(replay_buffer_filename,'wb') as wfp:
                    pickle.dump(buff, wfp)
                # Re-load our database
                with open(replay_buffer_filename,'rb') as rfp:
                    buff = pickle.load(rfp)

            if i == save_frequency:
                #print configuration information as text file in ddpg_summary
                print("Saving training configuration text file...")
                with open("ddpg_summary/" + current_date_time + "/Config.txt", "w") as text_file:
                    line1 = "NOISE_MAX_EP: {0} \n".format(NOISE_MAX_EP)
                    line2 = "map_name: {0} \n".format(map_name)
                    line3 = "prevent_PID: {0} \n".format(prevent_PID)
                    line4 = "save_frequency: {0} \n".format(save_frequency)
                    line5 = "LASER_BEAM: {0} \n".format(LASER_BEAM)
                    line6 = "save_velocity_bool: {0} \n".format(save_velocity_bool)
                    line7 = "load_replay_buffer_bool: {0} \n".format(load_replay_buffer_bool)
                    line8 = "current_date_time: {0} \n".format(current_date_time) 
                    line9 = "config_notes: {0} \n".format(config_notes)
                    text_file.writelines([line1, line2, line3, line4, line5, line6, line7, line8, line9])

                with open("saved_networks/Config.txt", "w") as text_file:
                    line1 = "NOISE_MAX_EP: {0} \n".format(NOISE_MAX_EP)
                    line2 = "map_name: {0} \n".format(map_name)
                    line3 = "prevent_PID: {0} \n".format(prevent_PID)
                    line4 = "save_frequency: {0} \n".format(save_frequency)
                    line5 = "LASER_BEAM: {0} \n".format(LASER_BEAM)
                    line6 = "save_velocity_bool: {0} \n".format(save_velocity_bool)
                    line7 = "load_replay_buffer_bool: {0} \n".format(load_replay_buffer_bool)
                    line8 = "current_date_time: {0} \n".format(current_date_time) 
                    line9 = "config_notes: {0} \n".format(config_notes)
                    text_file.writelines([line1, line2, line3, line4, line5, line6, line7, line8, line9])

            print("Save complete")

        print '| Reward: %.2f' % ep_reward, " | Episode:", i, \
        '| Qmax: %.4f' % (ep_ave_max_q / float(j)), \
        " | LoopTime: %.4f" % (np.mean(loop_time_buf)), " | Step:", j-1, '\n'
            


def main(_):

    with tf.compat.v1.Session() as sess:
        env = StageWorld(LASER_BEAM, map_type)
        np.random.seed(RANDOM_SEED)
        tf.compat.v1.set_random_seed(RANDOM_SEED)

        state_dim = LASER_BEAM * LASER_HIST + TARGET + (SPEED + ACCEL + JERK) * LASER_HIST

        action_dim = ACTION
        #action_bound = [0.25, np.pi/6] #bounded acceleration
        action_bound = [0.5, np.pi/3] #bounded velocity
        switch_dim = SWITCH

        discrete = False
        print('Continuous Action Space')

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, 
            ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim, switch_dim, 
            CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
        reward = Reward(REWARD_FACTOR, GAMMA)

        try:
            train(sess, env, actor, critic, noise, reward, discrete, action_bound)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    tf.compat.v1.app.run()
