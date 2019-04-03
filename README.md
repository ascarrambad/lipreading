# Exploring domain adaptation for lipreading

## Brief description of the scope and objectives of the thesis
Nowadays building a machine learning model that masters the problem of lipreading is
a hard task. In order to train a model that would provide a decent accuracy on
never before seen speakers, a very large amount of data is required. This is
due to the different physiological factors and articulatory patterns in the shape
of the mouth of different speakers.

Thus, this thesis will explore fully trainable deep learning methods capable of
extracting features that are speaker invariant, that is invariant w.r.t. shifts
in the data distribution between videos of different speakers. While there is no
clear path to solve this problem, we will experiment on separately
learning the motion dynamics and speech content, on specific training objectives
(in the style of domain-adversarial training), and/or on fast adaptation paradigms.
