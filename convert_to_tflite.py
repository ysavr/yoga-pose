import tensorflow as tf

# Path ke model .h5 Anda
h5_model_path = 'pose_classifier.h5'

# Path untuk menyimpan model .tflite
tflite_model_path = 'pose_classifier.tflite'

# Load model .h5
model = tf.keras.models.load_model(h5_model_path)

# Konversi model ke format TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Opsional: Optimasi untuk ukuran atau kecepatan (pilih salah satu atau beberapa)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimasi default
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] # Optimasi ukuran
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY] # Optimasi kecepatan

# Opsional:  Jika Anda menggunakan input/output float16
# converter.target_spec.supported_types = [tf.float16]

# Opsional: Jika Anda ingin menentukan input shape secara eksplisit (disarankan)
# Dapatkan input shape dari model Anda dan masukkan di sini
input_shape = model.layers[0].input_shape  # Ambil input shape layer pertama
input_tensor_name = model.layers[0].input.name
# Misalnya, jika input shape Anda (1, 224, 224, 3)
converter.input_shape = input_shape
converter.input_arrays = [input_tensor_name]

# Opsional: Jika Anda memiliki beberapa input
# converter.input_shapes = {'input_1': [1, 224, 224, 3], 'input_2': [1, 64, 64, 1]}
# converter.input_arrays = ['input_1', 'input_2']

# Konversi
tflite_model = converter.convert()

# Simpan model .tflite
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model berhasil dikonversi dan disimpan di: {tflite_model_path}")


# --- VERIFIKASI (Opsional) ---
# Memastikan model .tflite dapat dibaca

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

print("Model .tflite berhasil di load dan tensor berhasil dialokasikan.")

# ---  INFO Model (Opsional) ---
# Menampilkan info input dan output tensor

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Details:", input_details)
print("Output Details:", output_details)