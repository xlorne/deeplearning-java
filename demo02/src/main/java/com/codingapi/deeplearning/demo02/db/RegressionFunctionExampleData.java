package com.codingapi.deeplearning.demo02.db;

import com.codingapi.deeplearning.demo02.learn.RegressionFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;

/**
 * 数据操作工具类
 * @author lorne
 * @date 2019-10-22
 * @description y = ax+b
 */
@Component
public class RegressionFunctionExampleData {

    private String filePath = "init/data.bin";

    public void randData(int number,double a,double b){
        RegressionFunction regressionFunction = new RegressionFunction(a,b);
        File file = new File(filePath);
        INDArray array =  Nd4j.create(number,3);
        for(int i=0;i<number;i++){
            //为什么将x/number ,是为了梯度下降可更高效的操作，而做的特征缩放
            BigDecimal x = new BigDecimal(i).divide(new BigDecimal(number),5,BigDecimal.ROUND_UP);
            BigDecimal y =  regressionFunction.getY(x);
            double[] row = new double[]{1,x.doubleValue(),y.doubleValue()};
            array.putRow(i, Nd4j.create(row));
        }
        System.out.println(array);
        try {
            Nd4j.saveBinary(array,file);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public INDArray loadData(){
        try {
            return Nd4j.readBinary(new File(filePath));
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

}
