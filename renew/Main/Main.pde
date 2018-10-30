import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;



int n_input = 1;
int n_hidden = 20;
int n_output = 1;
double eta = 1.0;
double alpha = 0.9;
int count_train = 30000;
int count;


double[][] x;
double[][] y;

double[][] testX;
double[][] testY;

NeuralNetwork nn;
String path = "inputData.dat";

void read(String path){
  String[] line = loadStrings(path);
  x = new double[line.length][n_input];
  y = new double[line.length][n_output];
  for(int i = 0; i < line.length; i++){
    for(int j = 0; j < n_input; j ++){
      x[i][j] = Double.parseDouble(line[i].split("\t")[j]);
    }
    for(int j = 0; j < n_output; j++){
      y[i][j] = Double.parseDouble(line[i].split("\t")[j + n_input]);
    }
  }
}

void setup(){
  size(500,500);
  count = 0;
  read(path);
  
  //NeuralNetwork
  nn = new NeuralNetwork(n_input, n_hidden, n_output, eta, alpha);
  nn.setSeed(0);
  nn.createNetwork();
  nn.setTeacher(x, y);
  
  //grid for line
  testX = new double[width][1];
  testY = new double[width][1];
  for(int i = 0; i < width; i++){
    testX[i][0] = (double)i / width; //<>//
  }
  
  //Train Start.
  println("0 : e = " + nn.evaluation());
  //for(int i = 0; i < count_train; i++){
  //  nn.train();
  //  println((i+1) + " : e = " + nn.evaluation());
  //} 
}

void draw(){  
  if(count < count_train){
    nn.train();
    println((count + 1) + " : e = " + nn.evaluation());
    background(255);
    for(int i = 0; i < x.length; i++){
      stroke(0);
      noFill();
      ellipse((int)(x[i][0]*500), height - (int)(y[i][0]*500), 20, 20);
    }
    for(int i = 0; i < testX.length; i++){
      for(int j = 0; j < n_output;  j++){
        testY[i][j] = nn.forward(testX[i])[j].output();
      }
      stroke(255, 0, 0);
      fill(255, 0, 0);
      ellipse((int)(testX[i][0]*500), height - (int)(testY[i][0]*500), 3, 3);
      //point((int)(testX[i][0]*500), height - (int)(testY[i][0]*500));
    }
    count++;
  } //<>//
}
