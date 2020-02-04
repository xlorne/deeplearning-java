package com.codingapi.deeplearning.demo10.learn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author lorne
 * @date 2019-11-15
 * @description e^{z}\over {sum_k e^zk}
 */
public class SoftMaxActivation implements Activation {

    @Override
    public INDArray activation(INDArray data) {
        int length = data.rows();
        INDArray z = data.sub(Nd4j.max(data,1).reshape(length,1).broadcast(length,data.columns()));
        INDArray exp = Transforms.exp(z);
        //sum(exp)
        INDArray sum = Nd4j.sum(exp,1).reshape(length,1).broadcast(length,exp.columns());
        //exp/sum
        INDArray res =  exp.divi(sum);
        return res;
    }

    @Override
    public INDArray derivative(INDArray a) {
        int columns =  a.max(1).amaxNumber().intValue();
        long length = a.length();
        double values[] = new double[(int)length];
        for(int i=0;i<length;i++){
            if(i==columns){
                values[i] = a.getDouble(i)*(1-a.getDouble(i));
            }else{
                values[i] = -a.getDouble(columns)*a.getDouble(i);
            }
        }
        return Nd4j.create(values).reshape(a.shape());
    }
}
