package com.codingapi.deeplearning.demo08.learn;

import com.codingapi.deeplearning.demo08.learn.utils.SerializeUtils;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.codec.binary.Hex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@Slf4j
public class SerializeUtilsTest {

    public static void main(String[] args) throws Exception{
        INDArray test = Nd4j.create(new double[]{0,1,2,3,4});
        byte[] data =  SerializeUtils.serialize(test);
        System.out.println(Hex.encodeHexString(data));
        INDArray res = SerializeUtils.deserialize(data,INDArray.class);
        System.out.println(res);
    }
}
