"""
Generation of plots for the activation function trends.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.24
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-10.0, 10.0, 1000)

# Schwellwertfunktion
plt.title('Step')
plt.plot(x.numpy(), (numpy.sign(x.numpy()) +1) * 0.5)
plt.grid(True)
plt.xlabel(r'$z$')
plt.ylabel(r'$\sigma (z)$')
plt.xlim(-5, 5)
plt.ylim(-0.2, 1.2)
plt.show()

# ReLU
plt.title('ReLU')
plt.plot(x.numpy(), tf.nn.relu(x).numpy())
plt.grid(True)
plt.xlabel(r'$z$')
plt.ylabel(r'$\sigma (z)$')
plt.xlim(-5, 5)
plt.ylim(-0.2, 1.2)
plt.show()

# Sigmoid
plt.title('Sigmoid')
plt.plot(x.numpy(), tf.nn.sigmoid(x).numpy())
plt.grid(True)
plt.xlabel(r'$z$')
plt.ylabel(r'$\sigma (z)$')
plt.xlim(-5, 5)
plt.ylim(-0.2, 1.2)
plt.show()

# Tanh
plt.title('Tanh')
plt.plot(x.numpy(), tf.nn.tanh(x).numpy())
plt.grid(True)
plt.xlabel(r'$z$')
plt.ylabel(r'$\sigma (z)$')
plt.xlim(-5, 5)
plt.ylim(-1.2, 1.2)
plt.show()

#Leaky RelU
plt.title('Leaky ReLU')
plt.plot(x.numpy(), tf.nn.leaky_relu(x, alpha = 0.01).numpy())
plt.grid(True)
plt.xlabel(r'$z$')
plt.ylabel(r'$\sigma (z)$')
plt.xlim(-5, 5)
plt.ylim(-0.2, 1.2)
plt.show()

#ELU
plt.title('ELU')
plt.plot(x.numpy(), tf.nn.elu(x).numpy())
plt.grid(True)
plt.xlabel(r'$z$')
plt.ylabel(r'$\sigma (z)$')
plt.xlim(-5, 5)
plt.ylim(-2, 4)
plt.show()

