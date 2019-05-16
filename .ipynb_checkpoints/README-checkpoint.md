# Job & Resume Salary Predictor

<div>
<h3 align = "center">Table of Contents</h3>
<ol>
<li>Scrape the Data</li>
    <ul>
        <li><a href = "./get_data/get_the_data.ipynb">Web Scraper</a></li>
    </ul>
<li>Original</li>
    <ul>
      <li><a href = "./run_models/original/Data_Cleaning.ipynb">Data Cleaning</a></li>
      <li><a href = "./Models.ipynb">Modeling</a></li>
    </ul>
<li>Poly</li>
    <ul>
        <li><a href = "./run_models/poly/Data_Cleaning.ipynb">Data Cleaning</a></li>
        <li><a href = "./Models.ipynb">Modeling</a></li>
    </ul>
<li><a href = "./flask_app">Flask App</a></li>
    <ul>
        <li><a href = "./flask_app/main.py">main.py</a></li>
    </ul>
<li>Extras</li>
    <ul>
        <li><a href = "./run_models/Skill_Recommender.ipynb">Skill Recommender</a></li>
        <li><a href = "./run_models/Get_Coefficients.ipynb">Word Coefficients</a></li>
    </ul>
    </ol>
    </div>
<h2 align = "center">How much am I worth? </h2>
<h3 align = "center"><i>The hardest part of job hunting is when the recruiter turns to you and says<br> “So, what kind of salary are you looking for?”</i></h3>

Upon hearing this question, my heart dropped and I started to flip through the storybook of my life. I realized that when put on the spot, I only think of the negative parts of my life instead of all the great things that I have done. 

I know that we are all affected by our prior experiences, sometimes with more emphasis on the mistakes than successes. With this in mind, how can I objectively measure my experiences and skills? In other words- can I build a tool that will accurately quantify my achievements, skills and experience in a way that translates to a dollar amount?

This was a great idea! A model that can take in either a resume or a job posting as an input and the output would be a salary or salary range- all without me having to worry about the subjective nature of my experiences.

## [Original](./run_models/original)
### [Data Aqcuisition](./get_data)

I decided to use Google Jobs as my source of jobs.  This is because they aggregate their jobs from a bunch of other job platforms. Additionally, they post salary estimates from ‘Glassdoor”, “Built in NYC”, and “PayScale”. 

I downloaded a [chrome driver](http://chromedriver.chromium.org/) to be used with [Selenium](https://www.seleniumhq.org/). You can use [this blog post](https://towardsdatascience.com/web-scraping-using-selenium-python-8a60f4cf40ab) by [Atindra Bandi](https://towardsdatascience.com/@bandiatindra)to learn more about Selenium and how it can be used. 

After inspecting the Google job platform, I was able to identify which data I would pull from the job posting.
I wanted the:
- Title
- Company name
- Job posting
- Location
- Estimated salary (From Glassdoor or the others)

I started by creating a function to take an input of a search term and Selenium would open up chrome and open a Google job search using the search term given. [See the code here](./get_data/get_the_data.ipynb)

Now that we have the proper job data, we can continue to pull more jobs from different search words.
I created a [function](./get_data/get_the_data.ipynb#Manual-Function) that can pull up to 150 jobs at a time but it will require a little bit of manual scrolling and input. 

After running the function, you will be asked to enter a search word (no quotes required). 
Once the page is opened, scroll on the list of jobs until the page stops refreshing. 
Once you are done, go back to the script, and type a ‘y’ to proceed. 
The script will go through each job and pull the necessary data and it will return a [DataFrame](./get_data/Jobs/full_jobs_df.csv). 


After pulling over 6,000 jobs, it was time to go through the data and clean it up. 
Our data had 2 issues: 
    1. Only about 3,000 columns had estimated salary data 
    2. The estimated salary data was a range of values (EX: $75,000 — $120,000) 

### [Data Cleaning](./run_models/original/Data_Cleaning.ipynb)
In order to fix this, I dropped all rows that did not have a salary, and I created a function to go through each cell in the salary column and find the average salary for the range (EX: \$75,000 - \$120,000 = \$97,500). 
Once it got all of the averages, I created a new column in the DataFrame to store all of these new numbers. 

In order to run text through a statistic model, we would have to turn our text into numbers. 
The cleaning technique that I used for the text consisted of tokenizing, lemmatizing, removing stop words, and vectorizing. 
I decided to use SkLearn’s TFIDF-Vectorizer to turn the text into numbers based on the frequency of the word in the document, and the entire corpus (all of our documents).

The hyper-parameters that I decided to use for the vectorizer was:

    - n_grams = (1,3)
    - max_df = .85
    - min_df = .15 
    - binary = True

### [Statistical Modeling](./run_models/original/Models.ipynb)

Now that the data was ready to be put through a stats model, I performed a train_test_split on it and initially ran it through a LinearRegression.
This was to get a “base score” that I can use to see the accuracy of my other models, and for inference.
I winded up using Ridge, Lasso, Random Forest, Gradient Boost, and even a Neural Network.

For all of these models, my metric was ```Root Mean Square Error```. 

```Glassdoor’s normal range``` ± $32,214 
      
```Linear Regression``` ±  $30,595 

```Lasso``` ± $30,458 
      
```Ridge``` ± $30,560 

```Random Forest``` ± $19,290 

```Gradient Boost``` ± $18,379 
      
```Neural Network``` ± $28,736

The model predicts salaries with a lower error that Glassdoor! 

## [Feature Engineered](./run_models/poly) [(Poly)](./run_models/poly)
My next step was to do some feature engineering and feature selection.

I made a [new notebook](./run_models/Data_Cleaning.ipynb) to keep this separated from my [original work](./run_models/original)
### [Data Cleaning](./run_models/poly/Data_Cleaning.ipynb)
In addition to what was done in the [data acquisition process before](#Original), I used **Sklearn's** Polynomial Features. This multiplies each column by every other column including itself. This helps becuase words can have a different meaning when they are used with other words. 

Becuase the words were being multiplied, I realized that the vectorizer would need to be tweeked and the ```Binary``` hyper-paramter would need to be set to ```False```

After performing this, we ended up with over **\$39,000** features ($198 ^2$ ). This is a lot of features to throw into a model so I turned to **Sklearn's** Principal Component Analysis (PCA) function. This technique allows you to select the features that are important while still keeping 95% of variance. After running our **39,000** features through a PCA, we came out with **964** features.

### [Statistical Modeling](./run_models/poly/Models.ipynb)

Like before, now that the data was ready to be put through a stats model, I performed a train_test_split on it and ran it through all of 6 of the models.

```Glassdoor’s normal range``` ± $ 32,214 
      
```Linear Regression``` ±  $ 25,390 

```Lasso``` ± $27,704 
      
```Ridge```  ± $24,862 

```Random Forest``` ± $18,057 

```Gradient Boost``` ± $18,033 
      
```Neural Network``` ± $15,194





### [Flask App](./flask_app)
I worked on a Flask App that allows a user to get their estimated salary in 3 easy steps.
  1. Either input text or drop in a resume/job posting (```.txt, .pdf, or .docx```). 
  2. Choose a model that you would like be used on your text.
  3. Press **SUBMIT**
  
The app takes the text, prepares the text for the model, runs it through the desired model, and outputs an estimated salary and a range ( Salary ± RMSE ).


I then decided to place my resume inside of the model to see what it predicted. 
<div>
<table border="1" class="dataframe">  
<thead> <tr style="text-align: center;"> <th> </th> 
<th>Linear Regression</th> <th>Lasso</th> <th>Ridge</th> <th>Random Forest</th> <th>Gradient Boost</th> 
<th>Neural Net</th> <th>Linear Regression Poly</th> <th>Lasso Poly</th><th>Ridge Poly</th> 
<th>Random Forest Poly</th> <th>Gradient Boost Poly</th> <th>Neural Net Poly</th> 
</tr> </thead> 

<tbody> <tr style="text-align: center;">      
<th>margin</th> 
<td>29,566</td> <td>30,009</td> <td>29,512</td> <td>19,087</td> <td>19,043</td> <td>27,505</td> 
<td>25390</td> <td>27,703</td> <td>24,861</td> <td>18,056</td>  <td>18,032</td> <td>15,193</td> 
</tr>

<tr style="text-align: center;"> 
<th>worth</th> 
<td>84,534</td> <td>82,372</td> <td>78,737</td> <td>95,026</td> <td>97,914</td> <td>93,068</td>
<td>87,647</td> <td>76,896</td> <td>82,104</td> <td>116,140</td> <td>115,486</td> <td>87,074</td>
</tr> </tbody>
</table>
</div>
This is a pretty accurate estimation for entry level jobs as a Data Scientist / Data Analyst in New York City. 

## Conclusion
As you can see, the ```Root Mean Squared Error``` for the models are over 18% better scores than our [original work](#Original) and over 33% better than Glassdoor's! 

I am happy with how this project turned out and I see a lot of room to grow this project.

Some of my ideas are:

- Running the model on other types of jobs (Not just tech)
- Having the flask app output suggestions such as:
  - Suggested job title based on your resume.
  - Suggested words to use in your resume.
  - Suggested job postings that best fit your resume.
  - Suggested skills to acquire to make yourself more valuable.