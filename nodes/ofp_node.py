#!/usr/bin/env python

import rospy
import yaml
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ofp.ofp import OnlineFieldPerception  # adjust to your actual module path

class OFPNode:
    def __init__(self, config_path):
        # Load YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.image_topic = config['image_topic']
        self.output_topic = config['output_topic']
        self.vis_topic = config['vis_topic']
        self.checkpoint_path = config['checkpoint']
        embedding = config['embedding']
        gpu = config['gpu']

        # Load model
        self.model = OnlineFieldPerception(embedder=embedding, gpu=gpu)
        self.model.load(self.checkpoint_path)

        self.bridge = CvBridge()
        self.pub_trav = rospy.Publisher(self.output_topic, Image, queue_size=1)
        self.pub_vis = rospy.Publisher(self.vis_topic, Image, queue_size=1)

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            trav_img = self.model.predict(cv_img, embedding=None)
            vis_img = self.model.visualize(cv_img, trav_img)

            self.pub_trav.publish(self.bridge.cv2_to_imgmsg(trav_img, encoding='mono8'))
            self.pub_vis.publish(self.bridge.cv2_to_imgmsg(vis_img, encoding='bgr8'))

        except Exception as e:
            rospy.logerr("Image callback failed: %s", str(e))

def main():
    rospy.init_node('ofp_node')
    config_path = rospy.get_param('~config')
    OFPNode(config_path)
    rospy.spin()

if __name__ == '__main__':
    main()
