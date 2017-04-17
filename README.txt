/**
* @file README.txt
* @Description Project to learn/implement Behavioural learning for
  Udacity's nanodegree program
* @author Vikas MK
* @version 1.0
* @date 2017-04-08
*/

1) Used data set shared by Udacity ( around 6000 odd ) samples with Lenet
architecture. The network was able to move the car somewhat, but it had a left
turn bias and did not know how left to turn

2) I next switched to the NVIDIA model and trained with 1000 samples. The car
   was able to move forward in a wobbly soft of way moving mostly left and would
   go offcourse

3) Added information of left and right and center cameras along with a course
  correction metric of 0.2. Ie angle measurements of left camera had a steering
  correction which is positive as we want the camera to turn right at this
  point and the opposite for the right side. With these changes and running for
  about 12000 samples on PC. The car was able to negotiate the first turn about
  halfway and then failed. There was a bug in the course correction code. After
  fixing it, the car was able to negotiate the first curve. But still failed
  at the bridge as that data was not trained with. The bridge is a special case
  as it has black borders instead of the while/yellow lines that the network
  would have come to recognize as road edges

3)Next, I tried running the NVIDIA model with more samples on the CPU which failed
  because of running out of memory. I had about 38000 samples to contend with
  and I was forced to use a generator.

4) Using the generator, I was able to train the model on GPU. The model
performed fairly well on track 1. I left the car in autonomous mode for a couple
of hours and it was still driving when I checked on it.




