package com.codingapi.deeplearning.demo05.learn;

import java.util.ArrayList;
import java.util.List;

/**
 * @author lorne
 * @date 2019-10-31
 * @description
 */
public class SimpleNeuralNetworkLayerBuilder {

    private List<SimpleNeuralNetworkLayer> layers;

    private SimpleNeuralNetworkLayerBuilder(){
        this.layers = new ArrayList<>();
    }


    public static SimpleNeuralNetworkLayerBuilder build(){
        SimpleNeuralNetworkLayerBuilder layer =  new SimpleNeuralNetworkLayerBuilder();
        return layer;
    }

    public SimpleNeuralNetworkLayerBuilder addLayer(int in,int out){
        int index = layers.size();
        layers.add(new SimpleNeuralNetworkLayer(layers,in,out,index,false));
        return this;
    }

    public SimpleNeuralNetworkLayerBuilder outLayer(int in,int out){
        int index = layers.size();
        layers.add(new SimpleNeuralNetworkLayer(layers,in,out,index,true));
        return this;
    }

    public List<SimpleNeuralNetworkLayer> list(){
        return layers;
    }

    public int size(){
        return layers.size();
    }

    public SimpleNeuralNetworkLayer get(int index) {
        return layers.get(index);
    }
}
