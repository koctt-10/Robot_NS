import tensorflow as tf
from object_detection.utils import dataset_util 

def create_tf_example(image_path, annotations):
    # Загрузка изображения
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()

    # Создание записи TFRecord
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature([ann['xmin'] for ann in annotations]),
        'image/object/bbox/xmax': dataset_util.float_list_feature([ann['xmax'] for ann in annotations]),
        'image/object/bbox/ymin': dataset_util.float_list_feature([ann['ymin'] for ann in annotations]),
        'image/object/bbox/ymax': dataset_util.float_list_feature([ann['ymax'] for ann in annotations]),
        'image/object/class/text': dataset_util.bytes_list_feature([ann['class'].encode('utf8') for ann in annotations]),
        'image/object/class/label': dataset_util.int64_list_feature([ann['label'] for ann in annotations]),
    }))
    return tf_example

# Генерация TFRecord
def generate_tfrecord(output_path, image_paths, annotations_list):
    writer = tf.io.TFRecordWriter(output_path)
    for image_path, annotations in zip(image_paths, annotations_list):
        tf_example = create_tf_example(image_path, annotations)
        writer.write(tf_example.SerializeToString())
    writer.close()

# Пример использования
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
annotations_list = [
    [{'xmin': 0.1, 'xmax': 0.5, 'ymin': 0.2, 'ymax': 0.6, 'class': 'cube', 'label': 1}],
    [{'xmin': 0.3, 'xmax': 0.7, 'ymin': 0.4, 'ymax': 0.8, 'class': 'pipe', 'label': 2}]
]
generate_tfrecord('train.record', image_paths, annotations_list)