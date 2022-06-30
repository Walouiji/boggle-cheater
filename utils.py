import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ts_example():

  def __init__(self, batch_size = 10000, n_steps = 51, multiple =1): 
    self.batch_size = batch_size
    self.n_steps = n_steps
    self.multiple = multiple
    n_steps = n_steps + multiple
    freq1, freq2, offsets1, offsets2 = np.random.rand( 4, batch_size, 1) 
    time = np.linspace( 0, 1, n_steps) 
    series = 0.5 * np.sin(( time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin(( time - offsets2) * (freq2 * 20 + 20)) # + wave 2 
    series += 0.1 * (np.random.rand( batch_size, n_steps) - 0.5) # + noise 
    self.series = series[..., np.newaxis].astype( np.float32)

    

  def dataset(self):
    lvalid, ltest = round(self.batch_size*0.7),  round(self.batch_size*0.9)
      
    X_train, y_train = self.series[: lvalid, :self.n_steps -1], self.series[: lvalid,-self.multiple-1:-1]
    X_valid, y_valid = self.series[ lvalid: ltest, :self.n_steps -1], self.series[ lvalid: ltest, -self.multiple-1:-1] 
    X_test, y_test = self.series[ ltest:, :self.n_steps - 1], self.series[ ltest:, -self.multiple-1:-1]

    print( "Train dataset :" + str(X_train.shape[0] ) )
    print( "Valid dataset :" + str(X_valid.shape[0] ) + " Series size : "+ str(X_valid.shape[1] ) )
    print( "Test  dataset :" + str(X_test.shape[0] ) + " Series size : "+ str(X_test.shape[1] ) )
    print( "Series length : "+ str(X_train.shape[1] ))
    print( "Prediction length : " + str(y_train.shape[1] ) )


    return [(X_train, y_train),(X_valid, y_valid), (X_test, y_test)]

def _plot_series_(series,n_steps, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])


def plot_series(X,y,i):
    n_steps = X.shape[1] 
    _plot_series_(X[i, :, 0], n_steps,y[i, 0],  y_label=("$x(t)$"))
    plt.show()



def _plot_learning_curves_(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)


def plot_learning_curves(history):
    _plot_learning_curves_(history.history["loss"], history.history["val_loss"])
    plt.show()


def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    _plot_series_(X[0, :, 0],n_steps = n_steps )
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :], "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)
    plt.show()
    
    


def plot_fit_history( history):
    """

    :param history:
    :return:
    """
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.gca().set_xlim(0, len(history.epoch) - 1)
    plt.gca()
    plt.show()

def model_test_accurrancy( model, x_test, y_test ):
    """

    :param model: a trained model
    :param x_test
    :param y_test
    :return: metric 1 evaluation
    """
    eval = model.evaluate( x_test, y_test )
    res = "Pourcentage d'accuracy :" + str( round( eval[1] * 100, 2 ) ) + "%"
    print( res )
    return res
