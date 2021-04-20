from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint


def create_history(save_path_mode, model, epochs, batch_size, train_generator, validation_generator):
    # 模型保存格式默认是saved_model,可以自己定义更改原有类来保存hdf5
    ckpt = ModelCheckpoint(save_path_mode, monitor='val_loss', verbose=1,
                           save_best_only=True, save_weights_only=True)
    history = model.fit(x=train_generator,
                        batch_size=train_generator.n // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n // batch_size,
                        callbacks=[ckpt])
    return history


def model_fit_save(model_name, model, model_save_dir, epochs, *args):
    save_path_header = f'{model_save_dir}/{model_name}'
    save_path_mode = save_path_header + \
        '-{epoch:02d}-loss{loss:.5f}-acc{accuracy:.5g}.h5'
    history = create_history(save_path_mode, model, epochs, *args)
    model.save(f"{save_path_header}-{epochs}.h5")  # 保存最后的模型
    return history


def history_frame(history):
    '''history 转换为 pd.DataFrame
    '''
    df = pd.DataFrame.from_dict(history.history)
    df.index = history.epoch
    return df


def write_csv_result(history_frame, save_dir, model_name, suffix=''):
    name = f"{save_dir}/{model_name}{suffix}.csv"
    # df = history_frame(history)
    history_frame.to_csv(name)
