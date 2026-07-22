# 2025.09.29 HIT-xiaowangzi
# Controller for the BrainCo dexterous hand; interface mirrors the dex3 controller
import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, shared_memory, Array, Lock
from pygame.time import Clock

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_


class Brainco_Right_Hand_JointIndex(IntEnum):
    RightHandThumb = 0
    RightHandThumb_aux = 1
    RightHandIndex = 2
    RightHandMiddle = 3
    RightHandRing = 4
    RightHandPinky = 5

class Brainco_Left_Hand_JointIndex(IntEnum):
    LeftHandThumb = 6
    LeftHandThumb_aux = 7
    LeftHandIndex = 8
    LeftHandMiddle = 9
    LeftHandRing = 10
    LeftHandPinky = 11

Brainco_Num_Motors = 6
Brainco_Left_Cmd = "rt/brainco/left/cmd"
Brainco_Left_State = "rt/brainco/left/state"
Brainco_Right_Cmd = "rt/brainco/right/cmd"
Brainco_Right_State = "rt/brainco/right/state"


class BraincoController():
    def __init__(
        self, 
        fps: int = 100, 
    ):
        print("########################################")
        print("Initialize [Brainco_Controller]...")
        self.fps = fps

        # Shared Arrays for hands
        self.right_hand_state_array = Array('d', Brainco_Num_Motors, lock=True)
        self.left_hand_state_array  = Array('d', Brainco_Num_Motors, lock=True)  
        self.right_hand_action_array = Array('d', Brainco_Num_Motors, lock=True)
        self.left_hand_action_array  = Array('d', Brainco_Num_Motors, lock=True)
        
        # initialize shared arrays with default value
        for arr in [self.right_hand_action_array, self.left_hand_action_array]:
            for i in range(Brainco_Num_Motors):
                arr[i] = 1.0

        # initialize publisher and subscriber
        self.left_hand_action_publisher = ChannelPublisher(Brainco_Left_Cmd, MotorCmds_)
        self.left_hand_action_publisher.Init()
        self.right_hand_action_publisher = ChannelPublisher(Brainco_Right_Cmd, MotorCmds_)
        self.right_hand_action_publisher.Init()
        self.left_hand_state_subscriber = ChannelSubscriber(Brainco_Left_State, MotorStates_)
        self.left_hand_state_subscriber.Init()
        self.right_hand_state_subscriber = ChannelSubscriber(Brainco_Right_State, MotorStates_)
        self.right_hand_state_subscriber.Init()

        # initialize publisher thread
        # self._publish_action(self.fps)
        self.publish_action_thread = threading.Thread(target=self._publish_action, args=(self.fps,))
        self.publish_action_thread.daemon = True
        self.publish_action_thread.start()

        # initialize subscribe thread
        # self._subscribe_state(self.fps)
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_state, args=(self.fps,))
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.right_hand_state_array) or any(self.left_hand_state_array):
                break
            print(f"right hand: {any(self.right_hand_state_array)}, left hand: {any(self.left_hand_state_array)}")
            print("[Brainco_Controller] Waiting to subscribe dds...")
            time.sleep(1)
        print("Initialize [Brainco_Controller] OK!")
        print("########################################")

    def get_state(self):
        state = {
            'qpos': np.zeros(Brainco_Num_Motors * 2)
        }
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):   
            state['qpos'][id.value] = self.right_hand_state_array[idx]
        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):    
            state['qpos'][id.value] = self.left_hand_state_array[idx]
        return state

    def set_action(self, action):
        qpos = action['qpos']
        # Right hand is the first 6 indices, left hand is the last 6 indices
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):   
            self.right_hand_action_array[idx] = qpos[id.value]
        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):    
            self.left_hand_action_array[idx]  = qpos[id.value]

    def _subscribe_state(self, fps):
        clock = Clock()
        # frame_count = 0
        while True:
            left_hand_msg = self.left_hand_state_subscriber.Read()
            right_hand_msg = self.right_hand_state_subscriber.Read()
            if right_hand_msg is not None:
                for idx, id in enumerate(Brainco_Right_Hand_JointIndex):  
                    self.right_hand_state_array[idx] = right_hand_msg.states[idx].q

            if left_hand_msg is not None:
                for idx, id in enumerate(Brainco_Left_Hand_JointIndex):  
                    self.left_hand_state_array[idx] = left_hand_msg.states[idx].q
                  
 
            clock.tick(fps)
            # # TODO: debug
            # frame_count += 1
            # if frame_count % 1000 == 0:
            #     print(f"Brainco Subscribe State:")
            #     self.print_hand_info(self.right_hand_state_array, self.left_hand_state_array)

    def _publish_action(self, fps):
        clock = Clock()
        left_hand_msg = MotorCmds_()
        right_hand_msg = MotorCmds_()
        left_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
        right_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
        # frame_count = 0
        while True:
            for idx, id in enumerate(Brainco_Right_Hand_JointIndex):   
                right_hand_msg.cmds[idx].q = self.right_hand_action_array[idx]
                right_hand_msg.cmds[idx].dq = 1.0
            for idx, id in enumerate(Brainco_Left_Hand_JointIndex):    
                left_hand_msg.cmds[idx].q = self.left_hand_action_array[idx]
                left_hand_msg.cmds[idx].dq = 1.0
            self.left_hand_action_publisher.Write(left_hand_msg)
            self.right_hand_action_publisher.Write(right_hand_msg)
            clock.tick(fps)
            # # TODO: debug
            # frame_count += 1
            # if frame_count % 1000 == 0:
            #     print(f"Brainco Publish Action:")
            #     self.print_hand_info(self.right_hand_state_array, self.left_hand_state_array)

    def print_hand_info(self, right_hand_array, left_hand_array):
        hand_qpos = np.zeros(Brainco_Num_Motors * 2)
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):   
            hand_qpos[id.value] = right_hand_array[idx]
        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):    
            hand_qpos[id.value] = left_hand_array[idx]
        print(f"hand_qpos: {hand_qpos}")