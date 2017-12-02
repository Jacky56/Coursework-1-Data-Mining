package com.Datamining;

import java.util.*;
import java.util.concurrent.Callable;

import org.jblas.*;

public class CrossValidation implements Callable{

   private int KNN,step;
   private boolean weighted;

   private DoubleMatrix X_train;
   private DoubleMatrix y_train;

   private DoubleMatrix X_val;
   private DoubleMatrix y_val;

   private int Role;

   public static boolean Allow_testSet_to_be_Appended = false;


   public CrossValidation(DoubleMatrix Train,DoubleMatrix Validation,int Role,int[] removeFeatures, int KNN, int step,boolean weighted) {

			  this.KNN = KNN;
			  this.step = step;
			  this.Role = Role;
			  this.weighted = weighted;

			  //splits data into upon initialising
			  //training data
			  X_train = DataManager.Normalize(DataManager.GetFeatures(Train, removeFeatures));
			  y_train = DataManager.GetClass(Train, Role);

			  //validation data
			  X_val = DataManager.Normalize(DataManager.GetFeatures(Validation, removeFeatures));
			  y_val = DataManager.GetClass(Validation, Role);
   }



   //return function when invoked
   public Record[] call() {

			  //gets y prediction
			  DoubleMatrix[] y_pred = KNN(X_train,y_train,X_val,KNN,step,weighted);

			  //initiate record set
			  Record[] set = new Record[y_pred.length];

			  //retrieves unique classes from dataset
			  List<String> classType = new ArrayList<String>();
			  for(Map.Entry<String, Integer> role : DataManager.count.get(Role).entrySet()) {
						 classType.add(role.getKey());
			  }

			  //generate the confusionmat accuracy
			  for(int i = 0; i < y_pred.length; i++) {
						 int[][] confusionMat = confusionMat(y_pred[i],y_val, classType.size());
						 double accuracy = Accuracy(y_pred[i],y_val);
						 //puts it into record
						 int k = KNN <= step ? KNN -1 : (i * step) + 1;
						 set[i] = new Record(y_pred[i], k, accuracy, confusionMat,classType);
			  }
			  return set;
   }

   //the KNN function
   private DoubleMatrix[] KNN(DoubleMatrix X_train, DoubleMatrix y_train,DoubleMatrix X_test, int n_neighbors,int step , boolean weighted) {

			  DoubleMatrix X_trainIncrement = X_train;
			  DoubleMatrix y_trainIncrement = y_train;

			  DoubleMatrix[] y_pred = new DoubleMatrix[(int)Math.ceil((float)n_neighbors/step)];

			  for (int i = 0; i < y_pred.length; i ++) {
						 y_pred[i] = new DoubleMatrix(X_test.rows,1);
			  }

			  //iterates all test values
			  for(int test = 0; test < X_test.rows; test ++) {

						 //distance2 calculates euclidean/magnitude distance -> build your own if you want.
						 double magVal[] = new double[X_trainIncrement.rows];
						 for(int i = 0; i < magVal.length; i++) {
									magVal[i] = X_test.getRow(test).distance2(X_trainIncrement.getRow(i));
						 }

						 //sort -> convert into y class via indexing
						 int[] ArgSort = new DoubleMatrix(magVal).sortingPermutation();
						 List<Double> outcome = new ArrayList<Double>();

						 //initiate data set for counting
						 HashMap<Double, Double> vote = new HashMap<Double, Double>();
						 for(int neighbors = 0; neighbors < n_neighbors; neighbors++) {
									double classType = y_trainIncrement.get(ArgSort[neighbors]);

									//unweighted
									double value = 1;
									//weighted
									if(weighted) {
											   double dist  = magVal[ArgSort[neighbors]];
											   value = 1d/(1d + dist);
									}
									//count the class types
									vote.put(classType, vote.containsKey(classType) ? vote.get(classType) + value : value);
									//optimisation in speed
									if(neighbors % step == 0 && n_neighbors > step) {
											   outcome.add(getHighest(vote));
									}
						 }
						 //optimisation in speed
						 if(n_neighbors <= step) {
									outcome.add(getHighest(vote));
						 }
						 //append outcome of class values of each dinstinct set of parameters
						 for(int i = 0; i < y_pred.length; i++) {
									y_pred[i].put(test, outcome.get(i));
						 }


						 //appends evaluated row from test set to training set, elects classType by common value
						 if(Allow_testSet_to_be_Appended) {
									y_trainIncrement = DoubleMatrix.concatVertically(y_trainIncrement, new DoubleMatrix(new double[]{getCommon(outcome)}));
									X_trainIncrement = DoubleMatrix.concatVertically(X_trainIncrement, X_test.getRow(test));
						 }
			  }

			  return y_pred;
   }

   //testing, uses getHighest function
   private Double getCommon(List<Double> outcome) {
			  HashMap<Double,Double> common = new HashMap<Double,Double>();
			  for(Double value : outcome) {
						 common.put(value, common.containsKey(value) ? common.get(value) + 1 : 1);
			  }
			  return getHighest(common);
   }


   //Now you guys are really gonna help me with my reports :))))))))))))))))))))))))))))))))))))))))))))

   //returns the most common value out of the class types
   private Double getHighest(HashMap<Double, Double> countSet) {
			  Map.Entry<Double, Double> maxCount = null;
			  for(Map.Entry<Double, Double> role : countSet.entrySet()) {
						 if(maxCount == null || maxCount.getValue() <= role.getValue() ) {
									maxCount = role;
						 }
			  }
			  return maxCount.getKey();
   }



   //combines all the confusion matrices from records
   private static int[][] confusionMat(DoubleMatrix y_pred,DoubleMatrix y_test, int noClass) {

			  int[][] M = new int[noClass][noClass];
			  DoubleMatrix y = DoubleMatrix.concatHorizontally(y_pred, y_test);

			  for(int i = 0; i < y.rows; i++) {
						 M[(int)y.get(i,0)][(int)y.get(i,1)] += 1;
			  }

			  return  M;
   }

   //finds accuracy from real class types to predictive class types
   private static double Accuracy(DoubleMatrix y_pred,DoubleMatrix y_test) {

			  double counter =0;

			  for(int i = 0; i < y_test.rows; i++) {
						 if(y_test.get(i) == y_pred.get(i)) {
									counter ++;
						 }
			  }
			  return counter/ y_test.rows;
   }

}
