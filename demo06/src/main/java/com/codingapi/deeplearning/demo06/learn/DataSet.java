package com.codingapi.deeplearning.demo06.learn;

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

    /**
     * 训练数据 x,y
     */
    private INDArray x;
    private INDArray y;


    //加载数据
    public DataSet() throws IOException {
        String filePath = "init/lr_data.csv";
        INDArray data =  Nd4j.readNumpy(filePath,",");

        x = data.getColumns(0,1);
        y = data.getColumns(2,3);
    }

    //数据的input数量
    public int inputSize(){
        return x.columns();
    }
}
