package com.codingapi.deeplearning.demo03.db;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;

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
        File file = new File(filePath);
        INDArray array =  Nd4j.create(number,3);
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
