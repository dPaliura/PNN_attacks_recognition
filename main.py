from pandas import read_csv, DataFrame as df
import numpy as np

from pnn import PNN


if __name__ == '__main__':
    np.random.seed(42)
    try:
        train_set = np.array(read_csv("data/input/KDDTrain_procsd_redcd.csv"))
        test_set = np.array(read_csv("data/input/KDDTest_procsd.csv"))

        n = int(train_set.shape[1]-1)

        input("Press 'Enter to start'")

        train_set_in = train_set[:,0:n]
        train_set_out = train_set[:,n]
        
        test_set_in = test_set[:,0:n]
        test_set_out = test_set[:,n]

        pnn = PNN(train_set_in, train_set_out)

        print("\tRECOGNITION  (testing set)")
        test_recog = pnn.recognize(test_set_in, test_set_out).squeeze()

        testDF = df({
            "expected": test_set_out,
            "recognized": test_recog
        })
        testDF.to_csv("data/output/KDD_testing_pnn.csv")

    except Exception as e:
        print("Exception occured:\n{}".format(e))
    finally:
        input("Press 'Enter to quit'")