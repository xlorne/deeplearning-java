package com.codingapi.deeplearning.demo10.learn;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author lorne
 * @date 2020/10/10
 * @description
 */
@Slf4j
public class MlpNeuralNetworkTest {

    private static final int num_epochs = 64;

    private static final int num_inputs = 28;
    private static final int num_outputs = 10;

    private static final int num_hiddens =  64;

    private static final double lr = 0.001;


    private DataSetIterator mnistTrain = null;
    private DataSetIterator mnistTest =null;

    INDArray w1 =  null;
    INDArray b1 =  null;

    INDArray w2 = null;
    INDArray b2 = null;


    private void init() throws IOException {
        int rngSeed = 123;
        mnistTrain = new MnistDataSetIterator(num_epochs, true, rngSeed);
        mnistTest = new MnistDataSetIterator(num_epochs, false, rngSeed);

        w1 =  Nd4j.rand(num_inputs,num_hiddens);
        b1 =  Nd4j.ones(num_hiddens);

        w2 = Nd4j.rand(num_hiddens,num_outputs);
        b2 = Nd4j.ones(num_outputs);
    }

    private void train() throws IOException {

       this.init();


        while (mnistTrain.hasNext()) {
            DataSet batch = mnistTrain.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();

            for (int i=0;i<num_epochs;i++){
                INDArray feature = features.getRow(i).reshape(28,28);
                INDArray label = labels.getRow(i).reshape(1,10);
//
                log.info("feature.shape()=>{}",feature.shape());
                log.info("label.shape()=>{}",label.shape());

                SameDiff sameDiff = forward();

                Map<String, INDArray> placeholders = new HashMap<>();
                placeholders.put("label",label);
                placeholders.put("feature",feature);

                INDArray res =  sameDiff.output(placeholders,"outputs").get("outputs");
                log.info("forward res:{}",res);

                Map<String,INDArray> gradients =  sameDiff.calculateGradients(placeholders,"w1","b1","w2","b2");

                INDArray w1Grad =  gradients.get("w1");
                INDArray b1Grad =  gradients.get("b1");

                INDArray w2Grad =  gradients.get("w2");
                INDArray b2Grad =  gradients.get("b2");

                w1 = w1.sub(w1Grad.mul(lr));
                b1 = b1.sub(b1Grad.mul(lr));

                w2 = w2.sub(w2Grad.mul(lr));
                b2 = b2.sub(b2Grad.mul(lr));

            }
        }

    }

    public static void main(String[] args) throws IOException {
        new MlpNeuralNetworkTest().train();
    }


    private SameDiff forward(){
        SameDiff sameDiff = SameDiff.create();

        SDVariable w1Var = sameDiff.var("w1",DataType.FLOAT);
        SDVariable b1Var = sameDiff.var("b1",DataType.FLOAT);

        SDVariable w2Var = sameDiff.var("w2",DataType.FLOAT);
        SDVariable b2Var = sameDiff.var("b2",DataType.FLOAT);

        SDVariable featuresVar = sameDiff.placeHolder("feature", DataType.FLOAT);
        SDVariable labelsVar = sameDiff.placeHolder("label",DataType.FLOAT);



        SDVariable z1 = sameDiff.dot(featuresVar,w1Var).add(b1Var);
        SDVariable a1 = sameDiff.nn.relu(z1,1);


        SDVariable z2 = sameDiff.dot(a1,w2Var).add(b2Var);
        SDVariable a2 = sameDiff.nn.softmax(z2);

        sameDiff.loss.softmaxCrossEntropy("outputs",labelsVar,a2,null);

        sameDiff.associateArrayWithVariable(w1, w1Var);
        sameDiff.associateArrayWithVariable(b1, b1Var);
        sameDiff.associateArrayWithVariable(w2, w2Var);
        sameDiff.associateArrayWithVariable(b2, b2Var);

        return sameDiff;
    }

    public void testMseBackwards() {

        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 3;
        SDVariable input = sd.var("in", DataType.FLOAT, new long[]{minibatch, nOut});
        SDVariable label = sd.var("label", DataType.FLOAT, new long[]{minibatch, nOut});

        SDVariable diff = input.sub(label);
        SDVariable sqDiff = diff.mul(diff);
        SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);
        SDVariable avgMSE = sd.mean("loss", msePerEx, 0);

        INDArray inputArr = Nd4j.rand(DataType.FLOAT, minibatch, nOut);
        INDArray labelArr = Nd4j.rand(DataType.FLOAT, minibatch, nOut);

        sd.associateArrayWithVariable(inputArr, input);
        sd.associateArrayWithVariable(labelArr, label);

        INDArray result = avgMSE.eval();
        assertEquals(1, result.length());

        sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());


    }





}
