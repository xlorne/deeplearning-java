package com.codingapi.deeplearning.demo07.learn;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

@Slf4j
public class MnistTest {

    public static void main(String[] args) throws Exception{
        int batchSize = 1;
        int rngSeed = 123;

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        DataSet dataSet =  mnistTrain.next();
        INDArray features = dataSet.getFeatures();
        INDArray labels = dataSet.getLabels();
        log.info("features:{}",features);
        log.info("lables:{}",labels);
    }
}
