"""
This code imports necessary libraries and defines functions to 
perform exploratory data analysis on LinkedIn connection data.

It loads the data, cleans it, calculates summary statistics, 
visualizes the data through plots and graphs, and allows the user
to filter and analyze subsets of data.

Key functions and features:

  - File upload and data loading
  - Data cleaning and preprocessing 
  - Calculation of summary statistics
  - Filtering data by company and position
  - Visualizations: bar charts, treemaps, wordclouds, network graph
  - Downloading filtered data as CSV
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from nltk.stem import WordNetLemmatizer
import numpy as np
import statsmodels.api as sm
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import matplotlib.pyplot as plt 
import networkx as nx
from pyvis import network as net
import streamlit.components.v1 as components
warnings.filterwarnings("ignore")
import os
st.set_page_config(layout="wide")
selected = option_menu(
            menu_title=None,  # required
            options=["Instructions" ,"Dashboard"],  # required
            icons=[ "book","house"],  # optional
            menu_icon="cast",  # optional
            default_index=1,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding":"100px","padding": "0!important", "background-color": "#2D2E32"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    'color':"#71FCAA",
                    # "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
if selected == "Instructions":
    st.title(":bar_chart: LinkedIn Network Exploratory Data Analysis")
    st.write("""## How to use:
  1. you will need your LinkedIn Connections CSV file which can be downloaded by following these steps:
   - Click the Me icon at the top of your LinkedIn homepage on the desktop browser.
   - Select Settings & Privacy from the dropdown.
   - Click Data privacy on the left pane.
   - Under the How LinkedIn uses your data section, click Get a copy of your data.
   - Select Want something in particular? ...
   - Select Connections.

2. upload the **connections.csv** file to the Streamlit web app 
###### Note: the connections CSV file might take up to 10 minutes to be ready after requesting it using the steps above; you must refresh the LinkedIn browser for the download button to appear after waiting.
##### Congrats :partying_face: Now you are ready to go and see the exciting graphs and visuals
    
 ### This site's code can be accessed using [Github Link](https://github.com/michelhaj/Linkedin_DataAnalysis)
##### Created by Michel Al-haj """)
    
def maxMinValueDategetter(x,p,d,month=False):
    
    maxValue=x[x[p]==x[p].max()][p].values[0]
    minValue=x[x[p]==x[p].min()][p].values[0]
    if month==True:
        minDate=x[x[p]==x[p].min()][d].values[0]
        maxDate=x[x[p]==x[p].max()][d].values[0]
        return maxDate,minValue,maxValue,minDate
    else:
        maxDate=x[x[p]==x[p].max()][d].dt.date.values[0]
        return maxDate,minValue,maxValue



if selected == "Dashboard":
   

    # st.title(":bar_chart: LinkedIn Network Data Analysis")
    # st.markdown('<style>div.block-container{padding-top:1rem}</style>',unsafe_allow_html=True)
    uploaded_file=st.file_uploader(":file_folder: Upload your file",type=["csv","xlsx","txt","xls"])
    if uploaded_file is not None:
        filename=uploaded_file.name
        st.write(filename)
        try:
            df=pd.read_csv(uploaded_file,skiprows=3)
        except Exception as e:
            print(e)
            df=pd.read_excel(uploaded_file,skiprows=3)

        
    else:
        df=pd.read_csv("Connections_base.csv",skiprows=3)
        

    df.columns=df.columns.str.replace(" ","_").str.lower()
    col1,col2=st.columns((2))
    nrows,ncolumns=df.shape[0],df.shape[1]
    nwithemails=df[df.email_address.notnull()].shape[0]
    n_uniqecompanies=len(df.company.unique())
    n_uniquepositions=len(df.position.unique())
    print(df.email_address.isnull().sum())
        
    if df.email_address.isnull().sum()>=(nrows//2):
        df.drop(columns=["email_address"],inplace=True)
    df.dropna(axis="rows",inplace=True)
    df.connected_on=pd.to_datetime(df.connected_on)

    startdate=pd.to_datetime(df.connected_on).min()
    endtdate=pd.to_datetime(df.connected_on).max()

    with col1:
        date1=pd.to_datetime(st.date_input("Start Date",startdate))
        col1_sub_1, col2_sub_1 =col1.columns(2)
        col1_sub_1.metric(label=":male-technologist: Total Connections",value= f"{nrows}")
        col2_sub_1.metric(label=":incoming_envelope: Connection with Emails",value= f"{nwithemails}")
    
    with col2:
        date2=pd.to_datetime(st.date_input("End Date",endtdate))
        col1_sub_2, col2_sub_2 =col2.columns(2)
        col1_sub_2.metric(label=":office: Unique companies connections work with", value=n_uniqecompanies)
        col2_sub_2.metric("Unique positions of connections", n_uniquepositions)
    df=df[df.connected_on.between(date1,date2)].copy()
    # print(df.info)
    company_list=st.sidebar.multiselect("Pick a company",df.company.unique())
    if not company_list:
        df2=df.copy()
    else:
        df2=df[df.company.isin(company_list)]
    position_list=st.sidebar.multiselect("Pick a position",df2.position.unique())
    if not position_list:
        df3=df2.copy()
    else:
        df3=df2[df2.position.isin(position_list)]
    if not company_list and not position_list:
        filtered_df=df
    elif company_list and position_list:
        filtered_df=df[df.company.isin(company_list) & df2.position.isin(position_list)]
    elif company_list:
        filtered_df=df[df.company.isin(company_list)]
    else:
        filtered_df=df[df2.position.isin(position_list)]


    with col1:
        print(filtered_df.groupby(by="connected_on").count().reset_index())
        st.subheader("Connections Count by Date")
        data_by_dateDf=filtered_df.groupby(by="connected_on").count().reset_index()
        fig=px.scatter (data_by_dateDf,x="connected_on",y="position",marginal_y="violin",
           marginal_x="box", trendline="ols",color="position",labels={'position':'Connections number'},template="plotly")
        st.plotly_chart(fig,use_container_width=True,height=400)

        ##############################################
        #getting the the value of the date with most connection ,getting the the value of most connections
        dateWithMostConnections,numberOfConnectionOnBestDate_least,numberOfConnection_Most=maxMinValueDategetter(data_by_dateDf,"position","connected_on")
        
        # Fit an OLS model
        data_by_dateDf['connected_on_timestamp'] = data_by_dateDf['connected_on'].astype('int64') // 10**9  # Convert nanoseconds to seconds
        x = data_by_dateDf["connected_on_timestamp"]
        y = data_by_dateDf["position"]
        X = sm.add_constant(x)  # Add a constant to the predictor
        model = sm.OLS(y, X)
        results = model.fit()

        # Extract the slope
        slope = results.params[1]  # The coefficient for the "connected_on" variable
        
        connnections_by_date_markdown_text = f'<p style="font-size:16px;">Based on your data which shows only days that records at least one connection, there is a <a style="color: #588499; text-decoration:none;font-weight:bold;"> {"Down trend " if slope <0 else "up trend "}</a> in the number of connections. The Date with most Connections is <a style="color: #588499; text-decoration:none; font-weight:bold;">{dateWithMostConnections}</a> with <a style="color: #588499; text-decoration:none;font-weight:bold;">{numberOfConnection_Most}</a> connections made. The minimum number of connections recorded in one day is \
        <a style="color: #588499; text-decoration:none;font-weight:bold;">{numberOfConnectionOnBestDate_least}</a>.</p>'
        st.markdown(connnections_by_date_markdown_text, unsafe_allow_html=True)
    filtered_df["month_name"]=filtered_df.connected_on.dt.month_name()
    with col2:
        st.subheader("Connections Count by Month")
        df_by_month=filtered_df.groupby(by="month_name").count().reset_index().sort_values(by="connected_on",ascending=False)
        fig=px.bar(df_by_month,template="plotly",x="month_name",y="position",text="position",color="month_name",labels={'position':'Connections number'})
        st.plotly_chart(fig,use_container_width=True,height=300)
        #######################################
        #getting the the value of the month with most connection ,getting the the value of most connections by month
        maxMonth,minValue,maxValue,minMonth=maxMinValueDategetter(df_by_month,"position","month_name",True)
        avgValue=round(df_by_month.position.mean(),2)

        #######################################
        connnections_by_date_markdown_text = f'<p style="font-size:16px;">Based on your data which shows only days that records at least one connection, the average number of connections per month is <a style="color: #588499; text-decoration:none;font-weight:bold;"> {avgValue}</a>. The month with most Connections is <a style="color: #588499; text-decoration:none; font-weight:bold;">{maxMonth}</a> with <a style="color: #588499; text-decoration:none;font-weight:bold;">{maxValue}</a> connections made. The month with the least number of connections is\
        <a style="color: #588499; text-decoration:none;font-weight:bold;">{minMonth}</a> which recorded <a style="color: #588499; text-decoration:none;font-weight:bold;">{minValue}</a> connection .</p>'
        st.markdown(connnections_by_date_markdown_text, unsafe_allow_html=True)
    # def comp_count_func():
        
    cl1,cl2=st.columns(2)
    top_companies=(filtered_df.merge(filtered_df.company.value_counts(),on="company").sort_values(by="count",ascending=False).reset_index(drop=True))
    top_companies["occurences"]=1
    top_companies["used_in_treemap_company"]=top_companies.apply(lambda x: 1 if  x.company in top_companies.company.values[:len(top_companies.company.values)//5] else 0,axis=1)
    top_companies=top_companies[top_companies.used_in_treemap_company==1]
    # top_companies=filtered_df.groupby(by=["company",'position']).count().reset_index().sort_values(by="company",ascending=False).reset_index(drop=True)
    top_positions=filtered_df.groupby(by="position").count().reset_index().sort_values(by="connected_on",ascending=False).reset_index(drop=True)
    with cl1:
        st.subheader("Top Companies / Organizations in your Network")
        if len(top_companies)<5:
            st.write("No enough data to show")
        else:
            fig=px.treemap(top_companies,path=["company","position","first_name"],values="occurences",template="plotly")#, values="connected_on",template="ggplot2",labels={"connected_on":"count"})
            st.plotly_chart(fig,use_container_width=True,height=300)
            pass
    with cl2:
        
        st.subheader("Top Positions in Your Network")
        if len(top_positions)<5:
            st.write("No enough data to show")
        else:
            fig=px.treemap(top_positions[:(len(top_positions)//6)],path=["position","company"], values="company",template="plotly",labels={"company":"count"})
            st.plotly_chart(fig,use_container_width=True,height=300)

    # import plotly.figure_factory as ff
    st.subheader("Connections  Data Table")  
    with st.expander("Data_Table"):
        df_sample=filtered_df.drop(columns=["url"])
        st.write(df_sample)                                                          
    wc1,wc2=st.columns(2)

    with wc1:
        st.subheader(f"Most Frequent Names in Your Network" )
        x=' '.join(filtered_df.first_name.to_list())
        stopwords=STOPWORDS
        wc=WordCloud(background_color='white',stopwords=stopwords,height=1000,width=1500)
        wc.generate(x)
        fig, ax = plt.subplots()
        plt.imshow(wc,interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig)
    top_companies=filtered_df.company.value_counts().reset_index()
    filtered_top_companies=top_companies.loc[top_companies['count'] >=2]
    filtered_df["full_name"]=filtered_df.first_name+" "+filtered_df.last_name
    print(set([(i,j)for i, j in filtered_df[filtered_df.company=="PwC"][['full_name',"position"]].values]))
    with wc2:
        st.subheader(f"Most Frequent Names and Positions in Your Network by Compny" )
        g=nx.Graph()
        g.add_node("Me")
        for _,row in filtered_top_companies.iterrows():
            company=row['company']
            count=row['count']
            title=f"{company} -({count})"
            # positons=set(filtered_df[filtered_df.company==company][["position",'full_name']].values)
            # names=set(filtered_df[filtered_df.company==company]['full_name'])
            position1="\n.".join([f"{i} -- {j}" for i, j in filtered_df[filtered_df.company==company][['full_name',"position"]].values])
            hover_info=f"{company}\n {position1}"
            g.add_node(company,size=count*2,title=hover_info,color="#6C94A6",title_color="#184091", hover_color="#184091")
            g.add_edge("Me",company,color="#74613B")
        nt=net.Network(height='500px',width='100%',notebook=True,bgcolor="#F1EEEB")
        nt.from_nx(g)
        nt.save_graph("graph.html")
        HtmlFile = open("graph.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code,width=600,height=500)
    import plotly.graph_objects as go
    st.subheader("Recency of Connections")
    filtered_df["connection_age"]=(pd.to_datetime("today")-filtered_df.connected_on).dt.days
    ages_in_days=filtered_df.connection_age.values
    labels=[]
    for age in ages_in_days:
        if age > 365:
            years = int(age / 365)
            labels.append(f'{years} years')
        else:
            months = int(age / 30)
            labels.append(f'{months} months')

    # Create the scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df.full_name.values,
        y=ages_in_days,
        mode='markers',
        text=labels,
        marker=dict(
            size=7,
            color=ages_in_days,
        )))
    

    # Customize layout if needed
    fig.update_layout(
        xaxis=dict(title='Full Name of Connections'),
        yaxis=dict(title='Age (days)'),

    hoverlabel=dict(
        bgcolor="blue",
        font_size=16,
        font_family="Rockwell"
    )
)
    

    # filtered_df["connection_age"]=filtered_df["connection_age"].apply(lambda x: x//365 if x>=365 else x//30)
    # fig=px.scatter(filtered_df,x="full_name",y="connection_age",labels={'connection_age':'days',"full_name":"name"})
    st.plotly_chart(fig,use_container_width=True,height=300)


    l,r=st.columns(2)
    filtered_df["year"]=filtered_df.connected_on.dt.year
    network_by_year=filtered_df.groupby(by="year").count().reset_index().sort_values(by="connected_on",ascending=False).reset_index(drop=True)
    with l:
        st.subheader("Network Growth by Year")
        fig=px.histogram(filtered_df.connected_on,x="connected_on",nbins=len(filtered_df.year.unique()),template="gridon",labels={'connected_on':'count'})
        st.plotly_chart(fig,use_container_width=True,height=300)
    with r:
        st.subheader("Network Growth by Year Sample Data")
        with st.expander("connections_count_By_Year_Table"):
            st.write(network_by_year[["year",'connected_on']].rename(columns={"connected_on":"count"}).style.background_gradient(cmap='Blues'))
            csv=network_by_year.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Data",data=csv,file_name="connections_count_By_Year.csv",mime="txt/csv",help="Click here to Download the data as csv file")

    industry_mapping = {
        'Healthcare': [
            'general dental practitioner', 'trainee pharmacist', 'clinical dietitian', 
            'medical intern', 'clinical pharmacist'
        ],
        'Technology & Software': [
            'technical account manager', 'ios developer', 'full stack java developer',
            'information security officer', 'software developer', 'information technology help desk',
            'networks & security engineer', 'cyber security engineer', 'it data analyst',
            'software development engineer', 'ui ux designer', 'web developer', 'frontend developer',
            'database developer', 'cybersecurity associate', 'it support specialist', 'cloud engineer',
            'it technical support specialist', 'ios software engineer', 'devops & it engineer'
        ],
        'Human Resources': [
            'senior human resources specialist', 'talent acquisition manager', 
            'senior human resources consultant', 'senior talent and development associate', 
            'human resources analyst', 'human resources intern', 'human resources coordinator', 
            'human resources officer', 'human resources consultant', 'recruitment coordinator'
        ],
        'Business & Management': [
            'sales associate', 'business development department manager', 'business development executive',
            'business analyst', 'sales manager', 'business intelligence specialist',
            'business development coordinator', 'business development specialist', 
            'senior business analyst', 'business analyst | cyber risk advisory', 
            'business insights analyst', 'business development', 'associate business consultant',
            'associate product manager', 'vice president', 'senior partner', 
            'project management officer', 'project development manager', 'project coordinator',
            'junior project manager', 'procurement officer', 'supply planner', 'operations engineer',
            'logistics specialist', 'program manager', 'senior manager', 'senior consultant', 
            'management consultant', 'partner', 'investment manager', 'sales executive'
        ],
        'Marketing & Communications': [
            'communication and marketing consultant', 'digital marketing intern', 
            'senior marketing communications specialist', 'social media marketing specialist',
            'marketing executive', 'salesforce associate', 'marketing and communication', 
            'communication executive', 'social media coordinator', 'digital transformation'
        ],
        'Finance & Accounting': [
            'accountant', 'tax associate', 'financial controller', 
            'senior financial analyst', 'regional accountant', 'general accountant', 
            'senior auditor', 'audit associate', 'external auditor', 'finance manager',
            'accounts payable accountant', 'corporate customer representative',
            'asset liability management analyst', 'treasury and cash management analyst',
            'financial analyst'
        ],
        'Customer Service': [
            'customer service representative', 'customer care center representative', 
            'customer service officer', 'client services representative', 'customer support representative', 
            'customer experience executive', 'customer services agent', 'customer support team leader' 
        ],
        'Data Science & Analytics': [
            'data analyst', 'senior business intelligence analyst', 'advanced data analytics fellow',
            'data scientist', 'operations data analytics specialist', 'advanced data analytics', 
            'data transformation associate', 'data management analyst', 'data engineer',
            'machine learning engineer', 'data analysis intern', 'it data analytics consultant',
            'data analysis'
        ],
        'Legal': [
            'trainee lawyer', 'legal intern'
        ],
        'Academia & Education': [
            'private tutor', 'assistant professor', 'lecturer', 'associate professor',
            'teacher assistant', 'student'
        ],
        'Research & Development': [
            'research analyst', 'research trainee', 'research and product development',
            'clinical research associate', 'research assistant', 'product development manager',
            'phd student', 'researcher'
        ],
        'Quality Assurance & Control': [
            'quality assurance executive', 'quality assurance engineer',
            'quality assurance and quality control intern', 'quality control'
        ],
        'Sales & Retail': [
            'sales associate', 'retail sales associate', 'salesperson', 'sales director', 
            'sales and marketing manager'
        ],
    

        'Engineering & Technical': [
            
            'technical support engineer', 'graduate mechanical engineer', 
            'networks & security engineer', 'mechanical design engineer', 
            'technical sales engineer', 'technical account manager'
        ],
        'Logistics & Operations': [
            'logistics specialist', 'operations engineer', 'operations team member',
            'o&m coordinator and warehouse manager', 'junior underwriting operations assistant',
            'supply planning junior associate', 'operations data analytics specialist',
            'payment services coordinator', 'logistics coordinator'
        ],
        'Product Development & Management': [
            'product manager', 'associate product manager', 'product operations associate',
            'project management officer', 'project development manager',
            'project coordinator', 'program advisor associate',
            'program and products analyst', 'program manager', 'product development manager',
            'junior project manager', 'programmer analyst', 'jpm'
        ],
        'Audit & Compliance': [
            'audit associate', 'experienced associate', 'senior audit associate', 'audit intern',
            'external auditor', 'assurance associate', 'risk assurance', 'quality assurance officer',
            'assurance transformation associate', 'assurance associate ii', 'audit assurance',
            'it audit intern', 'risk & quality associate'
        ],
        'Information & Cyber Security': [
            'information security officer', 'cyber security engineer', 'information technology consultant',
            'information technology operations engineer', 'cybersecurity analyst',
            'information technology help desk senior'
        ],
        'Sales & Business Development': [
            'sales associate', 'sales manager', 'business development manager',
            'business development executive', 'regional sales manager',
            'business analyst', 'sales executive', 'business development specialist',
            'salesforce consultant', 'senior sales executive',
            'business development - vice president', 'senior business analyst'
        ],
        'Legal & Regulatory': [
            'legal intern', 'trainee lawyer', 'associate - grcs', 
            'risk & quality associate – ethics & compliance'
        ],
        'Education & Training': [
            'private tutor', 'assistant professor', 'lecturer', 
            'associate professor', 'teacher assistant', 'education consultant',
            'training specialist'
        ],
        # Add further mappings for other job titles as needed
    }
    # Function to classify industry
    # def classify_industry(position):
    #     for industry, keywords in industry_mapping.items():
    #         for keyword in keywords:
    #             if keyword in position.lower():
    #                 return industry
    #     return 'Other'
    # Create TF-IDF matrix
   
   

# print(filtered_df)


    industry_keywords_combined = {industry: ' '.join(keywords) for industry, keywords in industry_mapping.items()}

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the combined industry keywords
    industry_tfidf = vectorizer.fit_transform(industry_keywords_combined.values())
    print(industry_tfidf)
    # Function to classify industry
    def classify_industry_tfidf(position, industry_tfidf, vectorizer):
        position_vector = vectorizer.transform([position])
        cosine_similarities = cosine_similarity(position_vector, industry_tfidf).flatten()
        industry_scores = dict(zip(industry_keywords_combined.keys(), cosine_similarities))
        top_industry = max(industry_scores, key=industry_scores.get)
        return top_industry if industry_scores[top_industry] > 0 else 'Other'

    # Classify each position in filtered_df
    filtered_df['Industry'] = filtered_df['position'].apply(
        classify_industry_tfidf,
        args=(industry_tfidf, vectorizer)
    )

    print(filtered_df)


    # Apply classification
    # filtered_df['Industry'] = filtered_df['position'].apply(classify_industry)
    #################
    industry_counts = filtered_df['Industry'].value_counts().reset_index()
    industry_counts.columns = ['Industry', 'Count']

    # Plot the results
    
    fig = px.bar(industry_counts, x='Count', y='Industry', 
             title='Distribution of LinkedIn Connections by Industry Based on Positions based on TF-IDF and cosine similarity',
             labels={'Count': 'Number of Connections', 'Industry': 'Industry'},
             text="Count",
             orientation='h',  # Horizontal bar chart
             template='plotly')
    st.plotly_chart(fig,use_container_width=True,height=400)

    # Code to assign industries would remain the same as previous example
