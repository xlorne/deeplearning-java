package com.codingapi.deeplearning.demo06.learn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author lorne
 * @date 2019-11-15
 * @description e^{z}\over {sum_k e^zk}
 */
public class SoftMaxActivation implements Activation{

    @Override
    public INDArray forward(INDArray x, INDArray w, INDArray b) {
        int length = x.rows();
        //z = w.Tx+b
        INDArray z = x.mmul(w).add(b.broadcast(length, b.columns()));
        //z = z - max(z)
        z = z.sub(Nd4j.max(z,1).reshape(length,1).broadcast(length,z.columns()));
        //a = exp(z)
        INDArray exp = Transforms.exp(z);
        //sum(exp)
        INDArray sum = Nd4j.sum(exp,1).reshape(length,1).broadcast(length,exp.columns());
        //exp/sum
        INDArray res =  exp.divi(sum);
        return res;
    }


    @Override
    public INDArray back(INDArray a) {
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
