package com.codingapi.deeplearning.demo06.learn;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class DataSetTest {


    @Test
    public void batch() throws Exception{
        DataSet dataSet = new DataSet();
        DataSetScalingHelper scalingHelper = new DataSetScalingHelper(dataSet);
        scalingHelper.scalingSelf();
        for (int i=0;i<100;i++){
            DataSet batch =  dataSet.getBatch(15);
            System.out.println(batch.getX());
            System.out.println(batch.getY());
            batch.setY(batch.getX().add(batch.getY()));
        }
    }
}
