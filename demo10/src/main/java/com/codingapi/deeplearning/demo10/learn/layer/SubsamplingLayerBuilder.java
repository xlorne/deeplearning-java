package com.codingapi.deeplearning.demo10.learn.layer;

/**
 * @author lorne
 * @date 2020/1/29
 * @description
 */
public class SubsamplingLayerBuilder {


    private int[] kernelSizes;
    private int[] strides;

    protected SubsamplingLayerBuilder(){

    }

    public SubsamplingLayer build(){
        return new SubsamplingLayer(kernelSizes,strides);
    }

    public SubsamplingLayerBuilder kernelSize(int ... kernelSizes) {
        this.kernelSizes = kernelSizes;
        return this;
    }

    public SubsamplingLayerBuilder stride(int ... strides) {
        this.strides = strides;
        return this;
    }

}
