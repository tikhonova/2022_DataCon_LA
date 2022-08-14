![dataconla](https://user-images.githubusercontent.com/41370639/184544271-fde8fac2-45be-4a85-a6f1-849b6a3c92cb.png)

# Recording
[Link to recording](https://i.imgur.com/w2E5Gen.jpg)
# Customer Lifetime Value's review of a lifetime

Why LTV? This metric is a double-edged sword: you define it correctly, and it will help cut through the noise; make a mistake, and it may instead cut into your profits. The goal of the DataCon LA overview is to serve as your all-encompassing guide, alerting you of some of the most common pitfalls and equipping you with some of the least known approaches.

### Contents
- Deck
- Python script with 3 most popular survival models
	1. *Parametric* - relies on stats distribution, e.g. Weibull; a continuous probability distribution used to analyze life data, model failure times, and access product reliability.
	2. *Semi-parametric* - most well-known semi-parametric technique is Cox regression; involves utilizing not only the duration and the censorship variables but using additional data (Age, Gender, Salary, etc.) as covariates. We ‘regress’ these covariates against the duration variable.  
	3. *Non-parametric* - where Kaplan-Meier is the most common approach; most flexible, does not depend on any distribution, and measures central tendency with the median value.
- Dummy data for a contractual subscription business setting to run the script

### Libraries
- openpyxl
- numpy
- pandas
- matplotlib
- plotly
- lifelines


### Sources
- http://brucehardie.com/papers/rfm_clv_2005-02-16.pdf
- http://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf
- https://brucehardie.com/notes/004/ #BG/NBD workbook
- https://www.youtube.com/watch?v=guj2gVEEx4s 
- https://github.com/CamDavidsonPilon/lifetimes #check 'More Information' for ETSY deck
- https://www.linkedin.com/pulse/customer-lifetime-value-ltv-part-3-common-mistakes-risks-reynolds/
- https://www.arcalea.com/blog/calculating-lifetime-value-ltv
- https://medium.com/tradecraft-traction/4-mistakes-to-avoid-when-calculating-ltv-1ade13f90ec9
- https://github.com/CamDavidsonPilon/lifetimes
- https://en.wikipedia.org/wiki/Buy_Till_you_Die
- https://www.business2community.com/customer-experience/4-ways-to-measure-the-lifetime-value-of-your-customers-02338898
- https://blog.fusebill.com/2014/08/12/customer-lifetime-value
- http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf
- https://lifelines.readthedocs.io/en/latest/Survival%20Analysis%20intro.html
- https://boostedml.com/2018/11/when-should-you-use-non-parametric-parametric-and-semi-parametric-survival-analysis.html
