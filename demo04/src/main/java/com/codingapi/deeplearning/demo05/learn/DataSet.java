package com.codingapi.deeplearning.demo05.learn;

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


    public DataSet() throws IOException {
        String filePath = "init/lr_data.csv";
        INDArray data =  Nd4j.readNumpy(filePath,",");

        x = data.getColumns(0,1);
        y = data.getColumns(2);
    }
}
