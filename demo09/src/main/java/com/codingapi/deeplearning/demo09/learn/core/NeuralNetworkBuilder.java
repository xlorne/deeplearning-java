package com.codingapi.deeplearning.demo09.learn.core;

import com.codingapi.deeplearning.demo09.learn.layer.NeuralNetworkLayerBuilder;
import com.codingapi.deeplearning.demo09.learn.loss.LossFunction;
import com.codingapi.deeplearning.demo09.learn.utils.SerializeUtils;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * @author lorne
 * @date 2019/12/23
 * @description
 */
public class NeuralNetworkBuilder {

    public static NeuralNetworkBuilder builder(){
        return new NeuralNetworkBuilder();
    }

    private NeuralNetworkLayerBuilder builder;
    private double lambda;
    private double alpha;
    private int numEpochs;
    private long seed;
    private LossFunction lossFunction;

    private NeuralNetworkBuilder() {
        lambda = 0;
        alpha = 0.1;
        numEpochs = 10000;
        seed = 123;
    }

    public NeuralNetworkBuilder layers(NeuralNetworkLayerBuilder builder){
        this.builder = builder;
        return this;
    }

    public NeuralNetworkBuilder numEpochs(int numEpochs){
        this.numEpochs = numEpochs;
        return this;
    }

    public NeuralNetworkBuilder lambda(double lambda){
        this.lambda = lambda;
        return this;
    }

    public NeuralNetworkBuilder alpha(double alpha){
        this.alpha = alpha;
        return this;
    }

    public NeuralNetworkBuilder seed(long seed){
        this.seed = seed;
        return this;
    }

    public NeuralNetworkBuilder lossFunction(LossFunction lossFunction){
        this.lossFunction = lossFunction;
        return this;
    }

    public NeuralNetwork build(){
        return new NeuralNetwork(lambda,alpha,numEpochs,seed,builder,lossFunction);
    }


    /**
     * 加载模型
     * @param path  模型地址
     * @return  NeuralNetwork对象
     */
    public static NeuralNetwork load(String path) throws IOException {
        File file = new File(path);
        if(!file.exists()){
            throw new FileNotFoundException("not find file:"+path);
        }
        byte[] data =  FileUtils.readFileToByteArray(file);
        NeuralNetwork neuralNetwork =  SerializeUtils.deserialize(data,NeuralNetwork.class);
        return neuralNetwork;
    }

}
