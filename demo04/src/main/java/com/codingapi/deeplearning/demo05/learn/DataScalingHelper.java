package com.codingapi.deeplearning.demo05.learn;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 特征缩放
 * @author lorne
 * @date 2019-10-31
 * @description
 */
public class DataScalingHelper {

    private double max;
    private double min;
    private DataSet data;

    public DataScalingHelper(DataSet data) {
        this.data = data;
        this.max = data.getX().maxNumber().doubleValue();
        this.min = data.getY().maxNumber().doubleValue();
    }

    public void scaling(){
       INDArray array =  data.getX();
       array =  array.sub(min).div((max-min));
       data.setX(array);
    }

}
