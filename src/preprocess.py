import numpy as np

def preprocess():
    ##this is a script to preprocess the results of the reference solution calculation
    #we smooth the noise from the numerical calculation and multiply by the constant (-2)

    #input: raw data from the reference solution computation (ie )


    from scipy.ndimage import generic_filter,median
    import os 
    dirname = os.path.dirname(__file__) #fetches current location 
    refsolfile=dirname+"/refsolution/value_orig.csv"
    
    ##fetch the perturbative underdamped approximation data
    arr = np.loadtxt(refsolfile,
                        delimiter=",")  

    times = np.unique(arr[:,0])
    for t in range(len(times)):  
        filter_delta = 200
        valtemp = arr[np.where(arr[:,0]==times[t]),-2].reshape(5000) 
        qtemp = arr[np.where(arr[:,0]==times[t]),1].reshape(5000) 
        val_out = generic_filter(valtemp,median,filter_delta,mode="nearest")
        arr[np.where(arr[:,0]==times[t]),-2] = val_out
        dvout = generic_filter(np.gradient(val_out[filter_delta:-filter_delta],qtemp[filter_delta:-filter_delta]),median,filter_delta,mode="nearest") 
        arr[np.where(arr[:,0]==times[t]),-1] = -np.interp(qtemp,qtemp[filter_delta:-filter_delta],dvout,left=dvout[0],right=dvout[-1])

    np.savetxt(dirname+"/refsolution/value.csv",arr ,fmt="%10.6f",header = "t,q,v,du",delimiter=',')

    return 



if __name__ == "__main__":
  
  preprocess()
  