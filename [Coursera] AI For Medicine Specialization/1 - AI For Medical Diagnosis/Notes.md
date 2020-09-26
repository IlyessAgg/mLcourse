# Notes

> Notes of different teachings.

[TOC]

## Applications of computer vision to medical diagnosis

- Dermatology — `diagnosis and treatment of skin disorders.`

  *Example of application* : determine in a suspicious region of the skin whether a mole is **skin cancer** or not.

  ***Challenge*** : model bias towards particular skin color.

- Ophthalmology — `diagnosis and treatment of eye disorders.`

  *Example of application* : analyze **retinal fundus** images to detect a pathology such as **diabetic retinopathy** (DR).

  ***Challenge*** : data imbalance problem.

- Histopathology — `examination of tissues under the microscope.`

  *Example of application* : analyze scanned microscopic images of tissue, called **whole-slide images**, and determine the extent to which a cancer has spread.

  ***Challenge*** : images are very large and cannot be fed directly into an algorithm without breaking them down.

## Key Challenges

1. Class Imbalance

   > There's **not an equal number** of examples of `non-disease` and `disease` in medical datasets. 

   This is a reflection of the **prevalence**, or the frequency of disease in the real-world, where we see that there are a lot more examples of `normal` than of `mass`, especially if we're looking at X-rays of a healthy population. In a medical dataset, you might see **100 times** as many normal examples as mass examples.

   * **Loss** — `how far is the output probability from the desired label.`

     Normal ***cross-entropy loss*** on the $\boldsymbol i^{th}$ training data case: 
     $$
     \mathcal{L}_{cross-entropy}(x_i) = -(y_i \log(f(x_i)) + (1-y_i) \log(1-f(x_i)))
     $$
     where $\boldsymbol x_i$ and $\boldsymbol y_i$ are the input features and the label, and $\boldsymbol f(x_i)$ is the output of the model, i.e. the probability that it is positive. 

     Imbalance in data induces **imbalance in contribution to the loss**.

     > Solution : modify the loss function to weigh the normal and the mass classes differently.

      One way of doing this is by multiplying each example from each class by a class-specific weight factor, $\boldsymbol w_{pos}$ and $\boldsymbol w_{neg}$, so that the overall contribution of each class is the same. 

     To have this, we want 
     $$
     w_{pos} \times freq_{p} = w_{neg} \times freq_{n},
     $$
     which we can do simply by taking 
     $$
     w_{pos} = freq_{neg}  \\
     w_{neg} = freq_{pos}
     $$
     This way, we will be balancing the contribution of positive and negative labels.

     **Weighted Loss** :
     $$
     \mathcal{L}_{cross-entropy}^{w}(x) = - (w_{p} y \log(f(x)) + w_{n}(1-y) \log( 1 - f(x) ) )
     $$

   * **Resampling** — `resample or reconstruct data so that the classes are balanced.`

     *Example of application*

     |  Examples   | Re-Sampled  |
     | :---------: | :---------: |
     |  P1 Normal  |  P3 Normal  |
     |  P2 Normal  |  P6 Normal  |
     |  P3 Normal  |  P1 Normal  |
     | **P4 Mass** |  P8 Normal  |
     |  P5 Normal  | **P7 Mass** |
     |  P6 Normal  | **P4 Mass** |
     | **P7 Mass** | **P7 Mass** |
     |  P8 Normal  | **P4 Mass** |

     There are many variations of sampling approaches like **undersampling** the `normal` class or **oversampling** the `mass` class.

2. Multi-Task

   > Building a model to learn multiple related tasks by leveraging useful information among them.

   * **Multi-Label or Multi-Task Loss **— `sum of the losses over the multiple classes.`

     *Example with 3 classes* :
     $$
     L(X, y) = L(X, y_{mass}) + L(X, y_{pneumonia}) + L(X, y_{edema})
     $$
     We can also account for class imbalance by applying `weighted loss` on each of these classes. This time, we'd not only have a weight associated with just the positive and negative labels, but it's for the labels associated with a particular class of disease.
     $$
     w_{mass, pos} = freq_{mass, neg}  \\
     w_{mass, neg} = freq_{mass, pos}  \\
     ...
     $$

3. Dataset Size

   >  Architectures for medical images (**Inception-v3, ResNet-34, DenseNet, ResNeXt and EfficientNets**) are data hungry and benefit from the `millions of examples with labels` found in image classification datasets. On the other hand, we often have to work with small datasets.

   * **Pre-train** the network — `start from patterns that have been learned when solving a different problem.`

     *Early layers* of the network: `low level features` that can be generalized, like edge of an object.

     *Later layer* of network: `high level features`, like the head of a penguin

   * **Fine-tuning** the network  — `repurposing a pre-trained model for the medical task.`

     Most common design choices :

     * Fine-tune all the layers.
     * Freeze the shallow layers and fine-tune the later layers

## Metrics

* Accuracy — `proportion of the total examples that the model correctly classified.`
  $$
  \begin{align*}
  Accuracy &= (correctly \ classified \ examples) / (total \ examples)\\
  &= P(correct) \\
  &= P(correct \cap disease) + P(correct \cap normal) \\
  &= P(correct | disease) * P(disease) + P(correct | normal) * P(normal)
  \end{align*}
  $$
  $$P(disease)$$ is called the **prevalence** — `probability of a patient having disease in a population.`

* Sensitivity & Specificity — `if a patient has a disease (- is normal), what is the probability of the model predicting it as a positive(- negative).` 

  Also called **True Positive Rate** & **True Negative Rate**.

$$
Sensitivity = P(correct | disease) = P(+ | d) = \frac{\#(+ \ and \ d)}{\#(d)} \\
Specificity = P(correct | normal) = P(-|n) = \frac{\#(- \ and \ n)}{\#(d)}
$$

* Positive & Negative Predicted Value (**PPV** & **NPV**) —  `if the model predicts positive (- negative),  what is the probability the patient has a disease (- is normal).` 

$$
PPV = P(disease|correct) = P(d|+) = \frac{\#(+ \ and \ d)}{\#(+)} \\
NPV = P(normal|correct) = P(n|-) = \frac{\#(- \ and \ n)}{\#(-)}
$$

## Image segmentation

* MRI data — `an MRI sequence is a 3D volume.`

  Furthermore, an MRI example will be made up of multiple sequences, and thus will consist of `multiple 3D volumes`. We can combine these multiple 3D volumes  by treating them as ***different channels*** (like RGB) and combining them all to produce 1 channel.

  **Challenge** : sequences may not be aligned with each other.

  > A preprocessing approach that's often used to fix this is called **Image Registration** — `transforming the images so that they're aligned or registered to each other.`

* Segmentation — `process of defining the boundaries of various tissues.`

  There are two approaches to that process :

  * 2D: break up all 3D volumes to many 2D slices. The drawback is we might risk **losing important 3D context**.
  * 3D: break 3D volumes to many 3D subvolumes. We also might lose spatial context.

* Data Augmentation

  We need to transform both the `input` and the `output segmentation`. The transformation also have to be applied to the **whole 3D volume**.

* Loss

  Popular loss function for segmentation : **Soft Dice Loss** — `works well with imbalanced data.`
  $$
  \mathcal{L}_{Dice}(p, q) = 1 - \frac{2\times\sum_{i, j} p_{ij}q_{ij} + \epsilon}{\left(\sum_{i, j} p_{ij}^2 \right) + \left(\sum_{i, j} q_{ij}^2 \right) + \epsilon}
  $$

  * $\boldsymbol p$ is our predictions
  * $\boldsymbol q$ is the ground truth 
  * In practice each $\boldsymbol q_i$ will either be 0 or 1. 
  * $\boldsymbol\epsilon$ is a small number that is added to avoid division by zero.
