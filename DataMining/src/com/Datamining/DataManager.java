package com.Datamining;


import java.util.*;
import java.io.*;
import java.nio.file.*;

import org.jblas.*;

public final class DataManager {

	//fail?
	private static List<HashMap<String,Integer>> count = new ArrayList<HashMap<String,Integer>>();
	
	private static String regex = "";
	
	//encapsulation 
	public static DoubleMatrix GetFeatures(DoubleMatrix dataSet,int[] col) {
		DoubleMatrix X = new DoubleMatrix(dataSet.rows,0);
		for(int i = 0; i < dataSet.columns; i++) {
			for(int exception : col) {
				if(i != exception) {
					X = DoubleMatrix.concatHorizontally(X, dataSet.getColumn(i));
				}
			}
		}
		return X;
	}
	
	//encapsulation
	public static DoubleMatrix GetClass(DoubleMatrix dataSet,int col) {
		return dataSet.getColumn(col);
	}
	
	public static void SetRegex(String reg) {
		regex = reg;
	}	
	
	//read csv
	public static ArrayList<String[]> ReadCSV(String dir,boolean skip) {
		ArrayList<String[]> dataSet = new  ArrayList<String[]>();
		
		Path path = Paths.get(dir);
		
		try (BufferedReader reader = Files.newBufferedReader(path)){
			String line = null;
			
			String[] attributes = (reader.readLine()).split(",");
			
			for(String a : attributes) {
				count.add(new HashMap<String,Integer>());
			}
			
			while ((line = reader.readLine()) != null) {
				
				//will read next line if skip = true and line has regex value
				if(!(line.contains(regex) && skip)) {
					String[] value = line.split(",");
					
					dataSet.add(value);
					
					for(int i = 0; i < value.length; i++) {
						if(!value[i].equals(regex)) {
							count.get(i).put(value[i], count.get(i).containsKey(value[i]) ? count.get(i).get(value[i]) + 1 : 1);
						}
					}
				}
				
			}
		} catch (IOException e) {
			System.out.println("Bad Directory. Cannot find >>> '" + dir+ "'");
		}
		
		return dataSet;
	}
	
	
	
	
	//replace regex/Null value with common/mode value in column
	public static ArrayList<String[]> ReplaceMissingValues(ArrayList<String[]> dataSet) {
		ArrayList<String[]> returnSet = dataSet;
		
		for(int row = 0; row < returnSet.size(); row++) {
			for(int col = 0; col < returnSet.get(row).length; col ++) {
				
				if(returnSet.get(row)[col].equals(regex)) {
					returnSet.get(row)[col] = getCommon(count.get(col)).getKey();
				}
				
			}
		}
		
		return returnSet;
	}
	
	//!?!
	private static Map.Entry<String,Integer> getCommon(HashMap<String,Integer> col) {
		
		Map.Entry<String, Integer> maxEntry = null;

		for (Map.Entry<String, Integer> entry : col.entrySet()) {
		    if (maxEntry == null || entry.getValue() > maxEntry.getValue()) {
		        maxEntry = entry;
		    }
		}
		
		return maxEntry;
	}
	
	
	//get 'stratified' sampling : fake, roll 0 to 1 and added in if lower than sampleRatio
	public static ArrayList<String[]> Sample(ArrayList<String[]> dataSet, double sampleRatio) {
		ArrayList<String[]> returnSet = new ArrayList<String[]>();
		
		for(String[] row : dataSet) {
			if(Math.random() <= sampleRatio) {
				returnSet.add(row);
			}
		}
		return returnSet;
	}
	
	
	
	//no cell can be Null for this function
	//
	//Z-normalization Z = (X-u) /s
	public static DoubleMatrix Normalize(DoubleMatrix dataSet) {
		
		DoubleMatrix returnSet = new DoubleMatrix(dataSet.rows,dataSet.columns);
		DoubleMatrix rawSet = dataSet;
		
		for(int row = 0; row < dataSet.rows; row++) {
			DoubleMatrix X = rawSet.getRow(row);
			double u = rawSet.getRow(row).mean();
			double s = GetStd(rawSet.getRow(row));
			DoubleMatrix Z = (X.sub(u)).div(s);
			returnSet.putRow(row, Z);
		}
		return returnSet;
		//return dataSetToMatrix(ReplaceMissingValues(dataSet,("?")));
		//return dataSetToMatrix(dataSet);
	}
	
	public static double GetStd(DoubleMatrix s) {
		double u = s.mean();
		double v = 0;
		for(double i : s.toArray()) {
			v += Math.pow(i - u, 2);
		}
		return Math.sqrt(v/(s.toArray().length -1));
	}
	
	
	
	//convert string attributes into double
	//no cell can be Null for this 
	public static DoubleMatrix dataSetToMatrix(ArrayList<String[]> dataSet) {
		
		HashMap<String, Double> StringToNumber = new HashMap<String, Double>();
		
		for(HashMap<String,Integer> col : count) {
			Double val = 0d;
			for (Map.Entry<String, Integer> entry : col.entrySet()) {
				try {
					Double.valueOf(entry.getKey());
				} catch (NumberFormatException e) {
					if(!StringToNumber.containsKey(entry.getKey())) {
						StringToNumber.put(entry.getKey(), val);
						val ++;
					}
				}
			}
		}
		
		DoubleMatrix returnSet = new DoubleMatrix(dataSet.size(),dataSet.get(0).length);
		
		for(int row = 0; row < dataSet.size(); row++) {
			ArrayList<Double> conversion = new ArrayList<Double>();
			for(String col : dataSet.get(row)) {
				try {
					conversion.add(Double.valueOf(col));
				} catch(NumberFormatException e) {
					conversion.add(StringToNumber.get(col));
				} catch (Exception e) {
					System.out.println("Null values");
				}
			}
			//System.out.println(conversion);
			returnSet.putRow(row,new DoubleMatrix(conversion));
		}		
		return returnSet;
	}
	
	//split data set to sample set depending on column picked
	public static List<ArrayList<DoubleMatrix>> split(DoubleMatrix dataSet,int col) {
		HashMap<Integer, ArrayList<DoubleMatrix>> bins = new HashMap<Integer, ArrayList<DoubleMatrix>>();
		
		
		 for(int row = 0; row < dataSet.rows; row++) {
			 
			 int binNo = (int)dataSet.get(row,col);
			 
			 if(bins.containsKey(binNo)) {
				 bins.get(binNo).add(dataSet.getRow(row));
			 } else {
				 ArrayList<DoubleMatrix> temp = new ArrayList<DoubleMatrix>();
				 temp.add(dataSet.getRow(row));
				 bins.put(binNo,temp);
			 }
			 
		 }
		 
		 
		List<ArrayList<DoubleMatrix>> returnBins = new ArrayList<ArrayList<DoubleMatrix>>();
		for (ArrayList<DoubleMatrix> entry : bins.values()) {
			returnBins.add(entry);
			//bins.put(entry.getKey(), value)
		}
		
		return returnBins;
	}
	
	private static Map.Entry<String,Integer> qwe(HashMap<String,Integer> col) {
		
		Map.Entry<String, Integer> maxEntry = null;

		for (Map.Entry<String, Integer> entry : col.entrySet()) {
		    if (maxEntry == null || entry.getValue() > maxEntry.getValue()) {
		        maxEntry = entry;
		    }
		}
		
		return maxEntry;
	}
}
