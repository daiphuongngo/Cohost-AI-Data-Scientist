
DAX 
```
Total Bookings Today = CALCULATE(COUNT(nevada_reservation[id]), DAY(nevada_reservation[booked_date]) = DAY(TODAY()))
```

```M
Prev Month Bookings = CALCULATE(COUNT(nevada_reservation[id]), DATESINPERIOD(nevada_reservation[booked_date],TODAY(),-1,MONTH))
```

M
```
Prev Week Bookings = CALCULATE(COUNT(nevada_reservation[id]), DATESINPERIOD(nevada_reservation[booked_date],TODAY(),-7,DAY)
```
```
Prev Day Bookings = CALCULATE(DISTINCTCOUNT(nevada_reservation[id]), DATESINPERIOD(nevada_reservation[booked_date],TODAY(),-1,DAY))
```
```
Single Guest or Not = IF(nevada_reservation[guest_number] = 1, 1, 0)
```

```
Single Pair or Group = IF(nevada_reservation[guest_number] = 1, "Single", IF(nevada_reservation[guest_number] = 2, "Pair", "Group")
```
```
Occupancy Rate per Month:
Dates 2 = 
var FullCalendar = ADDCOLUMNS(CALENDAR("2015/1/1","2023/12/31"),"Month Number",MONTH([Date]),"Year",YEAR([Date]),"Year-Month",LEFT(FORMAT([Date],"yyyyMMdd"),6),"Month Name",FORMAT(MONTH([Date]),"MMM"),"MonthName-Year", FORMAT(MONTH([Date]),"MMM") & " " & YEAR([Date]))
return 
SUMMARIZE(FullCalendar,[Month Number],[Year],[Year-Month],[MonthName-Year])
```
```
Date = CALENDAR ( "2015-01-01", "2023-12-31" )
```
MonthInCalendar = FORMAT(MONTH([Date]),"MMM") & " " & YEAR([Date])

MonthInCalendar_4 = Dates[Month] & " " & YEAR(Dates[Year])


Occupied Days Within Month = 
VAR AdmitDate = VALUE( SELECTEDVALUE( nevada_reservation[check_in_date]) )
VAR DepartureDate = VALUE( SELECTEDVALUE( nevada_reservation[check_out_date]  ) )
VAR MinDateInContext = VALUE( MIN( Dates[Date] ) )
VAR MaxDateInContext = VALUE( MAX( Dates[Date] ) )

RETURN
IF( AND( AdmitDate < MinDateInContext, DepartureDate > MinDateInContext ) ,
        MIN( DepartureDate, MaxDateInContext ) - MinDateInContext,
            IF( AND( AND( AdmitDate > MinDateInContext, AdmitDate < MaxDateInContext ), DepartureDate > MinDateInContext ),
                 MIN( DepartureDate, MaxDateInContext ) - AdmitDate, 
                    BLANK() ) )


DaysinMonth = DAY( 
    IF(
        MONTH(Dates[Date_Copied]) = 12,
        DATE(YEAR(Dates[Date_Copied]) + 1,1,1),
        DATE(YEAR(Dates[Date_Copied]),  MONTH(Dates[Date_Copied]) + 1, 1)
    ) - 1
)

Revenue = SUM(nevada_reservation[earning])
Revenue_Measure = SUM(nevada_reservation[earning])

ADR = DIVIDE(SUMX(nevada_reservation, nevada_reservation[Revenue]),SUMX(nevada_reservation,nevada_reservation[nights]))

All Rooms = SUMX(nevada_reservation,nevada_reservation[rooms]) * SUMX(DISTINCT(Dates[DaysInMonth]), Dates[DaysInMonth])

Occupancy % = DIVIDE(SUMX(nevada_reservation, nevada_reservation[nights]),[All Rooms])

RevPAR = DIVIDE(SUMX(nevada_reservation, nevada_reservation[Revenue]),[All Rooms])

Pax / RN = sum(nevada_reservation[guest_number]) / sum(nevada_reservation[nights])   


YearMonthInt = FORMAT (Dates[Date], "YYYYMM")

Difference btw Booked and CI = DATEDIFF (nevada_reservation[booked_date], nevada_reservation[check_in_date], DAY )

Difference btw CI & Booked Date = Duration.Days([check_in_date] - [booked_date])
In Advance Booking = if [#"Difference btw CI & Booked Date"] >= 0 and [#"Difference btw CI & Booked Date"] <= 3 then "Within 3 Days" 
else if [#"Difference btw CI & Booked Date"] > 3 and [#"Difference btw CI & Booked Date"] <= 7 then "Within A Week"
else if [#"Difference btw CI & Booked Date"] > 7 and [#"Difference btw CI & Booked Date"] <= 31 then "Within A Month"
else if [#"Difference btw CI & Booked Date"] > 31 and [#"Difference btw CI & Booked Date"] <= 62 then "Within 2 Months"
else if [#"Difference btw CI & Booked Date"] < 0 then "Booked later than CI Date"
else "More than 2 Months"

LOS(MonthOfCheckIn) = IF(MONTH(nevada_reservation[check_in_date]) >= MONTH(nevada_reservation[check_out_date]), 
                        DATEDIFF(nevada_reservation[check_in_date], nevada_reservation[check_out_date], DAY), 
                        DATEDIFF(nevada_reservation[check_in_date], ENDOFMONTH(nevada_reservation[check_in_date]), DAY) + 1)

Check_Out_Start_of_Month = Date.StartOfMonth([check_out_date])

LOS(MonthOfCheckOut) = IF(MONTH(nevada_reservation[check_in_date]) < MONTH(nevada_reservation[check_out_date]), DATEDIFF(nevada_reservation[Check_Out_Start_of_Month], nevada_reservation[check_out_date], DAY), 0)


Due Date Table:
Due Date = CALENDARAUTO(6)

Due Fiscal Year = "FY " & if(MONTH('Due Date'[Due Date]) >=7 && MONTH('Due Date'[Due Date]) <=12,1) + YEAR('Due Date'[Due Date])

Due Fiscal Quarter = 
    var monthnumber=MONTH('Due Date'[Due Date])
    return 
        "Q " &
            SWITCH( 
                TRUE(),
                monthnumber >=1 && monthnumber <=3,3,
                monthnumber >=4 && monthnumber <=6,4,
                monthnumber >=7 && monthnumber <=9,1,
                monthnumber >=10 && monthnumber <=12,2
            )
            & " " 
            & 'Due Date'[Due Fiscal Year]

Due Month = FORMAT('Due Date'[Due Date],"mmmm, yyyy")

Due Full Date = FORMAT('Due Date'[Due Date],"yyyy mmmm, dd")

MonthKey = YEAR('Due Date'[Due Date]) * 100 + MONTH('Due Date'[Due Date])

Sales Table:

C Revenue = Sales[Sales Amount] - Sales[Total Product Cost]




```Javascript
var a = 20.
```