package com.codingapi.deeplearning.demo03.db;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

/**
 * @author lorne
 * @date 2019-10-31
 * @description 数据集
 */
@Data
public class DataSet {

    private INDArray x;
    private INDArray y;


    public DataSet(String filePath) throws IOException {
        INDArray data =  Nd4j.readNumpy(filePath,",");

        int length = data.columns();
        x = data.getColumns(0,length-2);
        y = data.getColumns(length-1);
    }

    public void print(){
        System.out.println(Nd4j.concat(1,x,y));
    }

    public void putX0() {
        x = Nd4j.concat(1,Nd4j.ones(x.rows(),x.columns()),x);
    }
}
