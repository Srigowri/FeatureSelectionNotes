Why do we need distance metrics:
We can start with using sample statistics to compare distributions. Comparing averages of the baseline and the distribution over the last minute.

Eg: Test distribution is anomalous if Avg(T) >> Avg(B)

If the averages are same, but the distribution is skewed then use 95th percentile between B and T to flag the anamoly.

If the averages are same, distribution is skewed and the 95th percentile is also the same, then we can distinguish between B and T(except visually)

Using a single sample statistics is not enough. Alernative is to use multiple sample statistics, statistical distances


**1. Kolmogorov-Smirnov test**
**2. Earth Mover's distance**
**3. Wassertein Distance**
**4. KL Divergence**
**5. Cramer distance**


https://www.youtube.com/watch?v=U7xdiGc7IRU
