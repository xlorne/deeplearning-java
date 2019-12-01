package com.codingapi.deeplearning.demo06.learn;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

/**
 * @author lorne
 * @date 2019-10-31
 * @description 数据集
 */
@Data
@Slf4j
public class DataSet {

    /**
     * 训练数据 x,y
     */
    private INDArray x;
    private INDArray y;

    private int count = 0;
    private int batchSize;

    //加载数据
    public DataSet(int batchSize) throws IOException {
        this.batchSize = batchSize;
        String filePath = "init/lr_data.csv";
        INDArray data =  Nd4j.readNumpy(filePath,",");

        x = data.getColumns(0,1);
        y = data.getColumns(2,3);
        count = x.rows();
    }


    private DataSet(INDArray x,INDArray y){
        this.x = x;
        this.y = y;
    }

    //数据的input数量
    public int inputSize(){
        return x.columns();
    }

    private int location = -1;

    private int[] rows(int batch){
        if(location==-1){
            location = 0;
        }
        if(location+batch>count){
            batch = count-location;
        }
        int [] rows = new int[batch];
        for(int i=0;i<batch;i++){
            rows[i] = location;
            if(location+1==count){
                location = 0;
                break;
            }
            location = location+1;
        }
        return rows;
    }

    private DataSet getBatch(int batch){
        int[] batchRows = rows(batch);
        INDArray x = getX().getRows(batchRows);
        INDArray y = getY().getRows(batchRows);
        return new DataSet(x,y);
    }

    public boolean hasNext() {
        if(location==0){
            location = -1;
            return false;
        }
        return true;
    }

    public DataSet next() {
        return getBatch(batchSize);
    }
}
