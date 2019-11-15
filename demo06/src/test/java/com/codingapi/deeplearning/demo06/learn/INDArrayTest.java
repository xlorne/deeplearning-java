package com.codingapi.deeplearning.demo06.learn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 * @author lorne
 * @date 2019-11-01
 * @description
 */
public class INDArrayTest {

    public static void main(String[] args) throws IOException {

        File file = new File("test.bin");

        DataOutputStream dataOutputStream = new DataOutputStream(new FileOutputStream(file));

        INDArray array1 = Nd4j.create(2,3);
        array1.putScalar(0,0,1);
        array1.putScalar(0,1,2);

        Nd4j.write(array1, dataOutputStream);

        INDArray array2 = Nd4j.create(2,3);
        array2.putScalar(0,0,3);
        array2.putScalar(0,1,4);

        Nd4j.write(array2, dataOutputStream);

        DataInputStream dataInputStream = new DataInputStream(new FileInputStream(file));
        INDArray res = Nd4j.read(dataInputStream);
        System.out.println(res);


    }
}
