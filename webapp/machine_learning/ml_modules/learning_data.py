import pandas


def import_train_data(public_data_frame_path, split=False):
    public_data_frame = pandas.read_csv(public_data_frame_path)
    if split:
        train = public_data_frame.truncate(after=50001)
        test = public_data_frame.truncate(before=50000)
        return train, test
    return public_data_frame
