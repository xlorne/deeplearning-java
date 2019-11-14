package com.codingapi.deeplearning.demo03.db;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;

/**
 * 数据操作工具类
 * @author lorne
 * @date 2019-10-22
 * @description y = ax+b
 */
@Component
@Slf4j
public class RegressionFunctionExampleData {

    private String filePath = "init/demo03.csv";

    public DataSet loadData(){
        DataSet dataSet = null;
        try {
            dataSet = new DataSet(filePath);
        } catch (IOException e) {
           log.error("load error",e);
        }
        return dataSet;
    }

}
