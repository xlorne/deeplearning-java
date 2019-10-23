package com.codingapi.deeplearning.demo01;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan(basePackages = "com.codingapi.deeplearning.demo01.mapper")
public class DeepLearningJavaDemo01Application {

    public static void main(String[] args) {
        SpringApplication.run(DeepLearningJavaDemo01Application.class, args);
    }

}
