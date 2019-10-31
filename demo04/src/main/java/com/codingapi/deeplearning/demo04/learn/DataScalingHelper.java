package com.codingapi.deeplearning.demo04.learn;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 特征缩放
 * @author lorne
 * @date 2019-10-31
 * @description
 */
public class DataScalingHelper {


    /**
     * x 的极值
     */
    private double Xmax;
    private double Xmin;

    /**
     * y 的极值
     */
    private double Ymax;
    private double Ymin;

    /**
     * 数据源
     */
    private DataSet data;


    /**
     * 记录极值
     */
    public DataScalingHelper(DataSet data) {
        this.data = data;
        this.Xmax = data.getX().maxNumber().doubleValue();
        this.Xmin = data.getX().minNumber().doubleValue();

        this.Ymax = data.getY().maxNumber().doubleValue();
        this.Ymin = data.getY().minNumber().doubleValue();
    }

    /**
     * 缩放数据
     */
    public void scaling(){
        INDArray x =  data.getX();
        x =  x.sub(Xmin).div((Xmax-Xmin));
        data.setX(x);

        INDArray y =  data.getY();
        y =  y.sub(Ymin).div((Ymax-Ymin));
        data.setY(y);
    }
}
