import preprocessing as preproc
import learning_data as ldata

public_data = 'twitter_data_public.csv'
private_data = 'twitter_data_private_nolabels.csv'

ltrain, ltest = ldata.import_data(public_data, split=True)
private = ldata.import_data(private_data)

ltrain, ltest, private = preproc.preprocess(ltrain), preproc.preprocess(ltest), preproc.preprocess(private)


