package com.codingapi.deeplearning.demo06.learn;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class DataSetTest {


    @Test
    public void batch() throws Exception{
        DataSet dataSet = new DataSet(13);
//        DataSetScalingHelper scalingHelper = new DataSetScalingHelper(dataSet);
//        scalingHelper.scalingSelf();
        for (int i=0;i<1;i++){
            while (dataSet.hasNext()){
                DataSet batch =  dataSet.next();
                System.out.println(batch.getX());
                System.out.println(batch.getY());
                batch.setY(batch.getX().add(batch.getY()));
            }

        }
    }
}
