from tensorflow.keras.layers import Layer, Input, Dense, Lambda, Flatten, Reshape, Concatenate, Conv2D, Conv2DTranspose # Añadir Concatenate
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model

class VAELossLayer(Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        original_input, reconstructed_output, z_mean, z_log_var = inputs
        
        input_dim = K.int_shape(original_input)[1] * K.int_shape(original_input)[2]

        reconstruction_loss = mse(K.flatten(original_input), K.flatten(reconstructed_output))
        reconstruction_loss *= float(input_dim) 
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss) # pérdida final
        self.add_loss(vae_loss)
        # Devuelve la salida reconstruida para que el modelo tenga una salida 'real'
        return reconstructed_output
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.0) # vector normal aleatorio
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# --- FUNCIÓN DE PLOTEO (Modificada para CVAE) ---
def plot_results_cvae(models, data, num_classes, batch_size=128, model_name="cvae_mnist"):
    encoder, decoder = models
    x_test, y_test_labels = data # y_test_labels son las etiquetas 0-9
    y_test_one_hot = to_categorical(y_test_labels, num_classes) # Convertir a one-hot para el encoder

    os.makedirs(model_name, exist_ok=True)

    # 1. Scatter plot del espacio latente (igual que antes, pero usando la etiqueta real para color)
    # El encoder ahora necesita la imagen Y la etiqueta one-hot
    z_mean_pred, _, _ = encoder.predict([x_test, y_test_one_hot], batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean_pred[:, 0], z_mean_pred[:, 1], c=y_test_labels, cmap="viridis", alpha=0.7) # Color por etiqueta 0-9
    plt.colorbar()
    plt.xlabel("z_mean[0]")
    plt.ylabel("z_mean[1]")
    plt.title("Espacio Latente (Media) - Coloreado por Dígito Real")
    plt.savefig(f"{model_name}/cvae_mean_scatter.png")
    plt.show()
    #plt.close()

    # 2. Generación Condicionada: Generar una fila por cada dígito
    n_examples_per_digit = 15 # Cuántos ejemplos generar por cada dígito
    digit_size = 28
    figure = np.zeros((digit_size * num_classes, digit_size * n_examples_per_digit))

    print(f"Generando {n_examples_per_digit} ejemplos para cada uno de los {num_classes} dígitos...")

    # Para cada dígito (0 a num_classes-1)
    for digit in range(num_classes):
        # Crear la etiqueta one-hot para este dígito, repetida n_examples_per_digit veces
        label_one_hot = np.zeros((n_examples_per_digit, num_classes))
        label_one_hot[:, digit] = 1

        # Muestrear puntos aleatorios del espacio latente (usando la distribución normal)
        # Podríamos también muestrear de un grid como antes, pero aleatorio es más simple aquí
        z_samples = np.random.normal(size=(n_examples_per_digit, latent_dim))

        # El decoder ahora necesita z Y la etiqueta one-hot
        x_decoded = decoder.predict([z_samples, label_one_hot], verbose=0)

        # Colocar los dígitos generados en la figura
        for i in range(n_examples_per_digit):
            img = x_decoded[i].reshape(digit_size, digit_size)
            figure[digit * digit_size: (digit + 1) * digit_size,
                   i * digit_size: (i + 1) * digit_size] = img

    plt.figure(figsize=(n_examples_per_digit * 0.8 , num_classes * 0.8)) # Ajustar tamaño
    plt.imshow(figure, cmap="Greys_r")
    plt.title("Dígitos Generados Condicionados por Clase (Filas)")
    plt.ylabel("Dígito Condicionado (0-9)")
    plt.xlabel("Ejemplos Generados")
    plt.yticks(np.arange(num_classes) * digit_size + digit_size/2, range(num_classes)) # Poner etiquetas de dígito en Y
    plt.xticks([]) # Quitar ticks en X
    plt.savefig(f"{model_name}/cvae_conditional_generation.png")
    plt.show()
    #plt.close()

# --- Carga y preprocesamiento de datos ---
(x_train, y_train_labels), (x_test, y_test_labels) = mnist.load_data() # y_train son labels 0-9
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1]).astype("float32") / 255.0
x_test = np.reshape(x_test, [-1, image_size, image_size, 1]).astype("float32") / 255.0

num_classes = 10
# Convertir etiquetas a one-hot para usarlas como input
y_train_one_hot = to_categorical(y_train_labels, num_classes)
y_test_one_hot = to_categorical(y_test_labels, num_classes)

# --- Parámetros del modelo ---
input_shape = (image_size, image_size, 1)
label_shape = (num_classes,) # Forma de la etiqueta one-hot
batch_size = 128
kernel_size = 3
filters = 16
latent_dim = 2 # Mantenemos 2D para visualización
epochs = 2 # CVAE puede necesitar un poco más de entrenamiento
model_name = "cvae_cnn"

# --- Encoder Modificado (Input: Imagen + Etiqueta) ---
image_inputs = Input(shape=input_shape, name='encoder_image_input')
label_inputs = Input(shape=label_shape, name='encoder_label_input') # añadimos esta entrada para la etiqueta one-hot

# Procesar imagen
x = image_inputs
for i in range(2):
    filters *= 2 
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
shape = K.int_shape(x)
x = Flatten()(x)

# Concatenar características aplanadas de la imagen con la etiqueta one-hot
concat_inputs = Concatenate()([x, label_inputs]) # añadimos 

# Capas Dense después de concatenar
x_concat  = Dense(128, activation='relu')(concat_inputs) # Capa intermedia más grande
z_mean    = Dense(latent_dim, name='z_mean')(x_concat)
z_log_var = Dense(latent_dim, name='z_log_var')(x_concat)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Encoder ahora toma lista de inputs
encoder = Model([image_inputs, label_inputs], [z_mean, z_log_var, z], name='encoder')
print("\nResumen del Encoder CVAE:")
encoder.summary()

# --- Decoder Modificado (Input: Vector Latente + Etiqueta) ---
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
label_inputs_decoder = Input(shape=label_shape, name='decoder_label_input')

# Concatenar vector latente z con la etiqueta one-hot al PRINCIPIO
concat_inputs_decoder = Concatenate()([latent_inputs, label_inputs_decoder])

# Mapear z concatenado a la forma pre-flatten del encoder
# Ajustar el tamaño de esta capa Dense 
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(concat_inputs_decoder)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Reconstruir con Conv2DTranspose (la lógica de filtros es la misma)
current_filters = shape[-1] # Filtros de la última capa Conv del encoder
for i in range(2):
    x = Conv2DTranspose(filters=current_filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
    current_filters //= 2

outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='decoder_output')(x)

# Decoder ahora toma lista de inputs
decoder = Model([latent_inputs, label_inputs_decoder], outputs, name='decoder')
print("\nResumen del Decoder CVAE:")
decoder.summary()

# --- CVAE (Conexión + Capa de Pérdida) ---
# Definir inputs principales para el modelo CVAE completo
main_image_inputs = Input(shape=input_shape, name='cvae_image_input')
main_label_inputs = Input(shape=label_shape, name='cvae_label_input')

# Obtener salidas del encoder
z_mean_out, z_log_var_out, z_out = encoder([main_image_inputs, main_label_inputs])
# Pasar z Y la etiqueta al decoder
outputs_cvae = decoder([z_out, main_label_inputs])

# La capa de pérdida necesita input original, reconstrucción, z_mean, z_log_var
# Nota: El input original es solo la imagen (main_image_inputs)
cvae_outputs = VAELossLayer()([main_image_inputs, outputs_cvae, z_mean_out, z_log_var_out])

# CVAE ahora toma lista de inputs
cvae = Model([main_image_inputs, main_label_inputs], cvae_outputs, name='cvae')

# --- Compilación y Entrenamiento ---
cvae.compile(optimizer='adam')
print("\nResumen del modelo CVAE completo:")
cvae.summary()

save_dir = "cvae_cnn_weights"
os.makedirs(save_dir, exist_ok=True)
weights_filename = f"cvae_cnn_mnist_{epochs}_epochs.weights.h5"
weights_path = os.path.join(save_dir, weights_filename)

print("\nIniciando entrenamiento del CVAE...")
# El entrenamiento ahora necesita pasar una LISTA de inputs: [imagenes, etiquetas_one_hot]
history = cvae.fit(
    [x_train, y_train_one_hot], # Lista de inputs de entrenamiento
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([x_test, y_test_one_hot], None), # Lista de inputs de validación
)

print(f"\nGuardando pesos entrenados en: {weights_path}")
cvae.save_weights(weights_path)

# --- Visualización de Resultados ---
models = (encoder, decoder)
data = (x_test, y_test_labels) # Pasar etiquetas numéricas para colorear el plot