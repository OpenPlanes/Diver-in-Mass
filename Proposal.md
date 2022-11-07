Summary
The Divebot is an autonomous dive float for recreational diving that combines traditional dive safety float with an underwater camera that follows divers. In its fully developed form it has the potential to solve a number of problem associated with dive sport and enhance the safety and enjoyment of diving activities.

These features include:

automated videography
accurate diver location signaling to surface vehicles
under water navigation
Air-Water wireless communication barrier
Phased implementation approach
Due to the constrained timeline of the EK505 class, the scope of this project will only aim to develop a scale proof of concept model, specifically to develop and test a camera tracking algorithm for underwater motion. Iterative phases of this project would build a full scale version, implement additional features and solve challenges that would potentially arise from environmental conditions. Below is the intended four phases for a fully developed system. Our project group will complete as many of these phases as time allows.

Phase 1 - Scale model, rigid boom and Camera tracking
Intended Features:

miniature version of robot
buoy tethered in place at surface
rigid boom from buoy to underwater camera with 2 degrees of freedom, one to raise and lower camera and one to rotate camera.
camera at bottom of boom
motion tracking of IR-light beacon attached to diver
Challenges to solve:

Geometry of robot
mass ratios of buoy, camera and boom
parsing camera data and writing algorithm to measure light intensity in image
robot controls based of camera image
waterproofing

![Phase1-sketch](https://user-images.githubusercontent.com/106100235/200344567-e2721e21-098f-4546-abfb-443b16838d94.jpeg)



Phase 2 - Scale model, cable line, mobile float, and camera tracking
Intended Features:

miniature version of robot
mobile surface buoy with 2 degrees of freedom to navigate "planar" surface of water
cable on winch connecting buoy to weighted camera pod to control camera depth and send signals. (1 degree of freedom)
camera pod with 1 degree of freedom to control axial rotation (about z-axis)
Challenges to solve:

geometry of cable as angle θ changes with drag underwater
positioning buoy based on light intensity of beacon.
Phase 2 Sketch

Phase 3 - Full size model, swimming pool test
Intended Features:

full size version of robot
radio transmission from buoy to surface receiver
adaptability to small wave perturbation (swimming pool conditions)
camera responds only to intended light source
Challenges to solve:

Radio transmission protocols and communication down cable
stability in water
better light tracking algorithm
Phase 4 - Full size model, open water test
Intended Features:

full size version of robot
Radio transmission from bouy to surface receiver
GPS receiver
adaptability to non-trivial waves (calm open water conditions)
Challenges to solve:
same as above but with harsher conditions


![Phase2-sketch](https://user-images.githubusercontent.com/106100235/200344645-91898237-086f-44c7-a79d-f8de5e5883eb.jpeg)


Image tracking intent and plan
To minimize cost of materials for initial research and to scale most basic version of project to the time available, we propose using a raspberry pi IR camera and using information based on the greyscale of the received images to determine the location and orientation of the robot in the environment, and then center itself on the source of light.

Infra-read light is used intentionally because of its poor propagation through water. If the dive bot remains close to a beacon attached to a diver, it will be the strongest source of IR light and in theory will be a bright spot in the image for the robot to lock onto. Other sources of light will emit less IR light and any light from the surface will be heavily attenuated. Once the robot receives and image and the intensity of the cells of light are measured it can determine how far above or below, and left or right of the centerline the light is, and adjust accordingly.

This project also aims to validate this IR light idea and determine a different source/color of light that would serves as a beacon, if this method is unsuccessful. The approach is also fairly simple and intentionally avoids complicated methods involving computer vision, image recognition and machine learning.
