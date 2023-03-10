RPTSCHED
	FIP WELLS /

RPTRST
	BASIC=6 FLOWS /

GRUPTREE
'GROUP1'	FIELD /
'GROUP2'        FIELD /
/

WELSPECS

  PROD1 'GROUP1' 2 2 1* OIL /
  INJ1  'GROUP2' 38 38 1* WATER /
  /

COMPDAT
  PROD1 25 10 1   41 OPEN 2* 0.152 3* Z//
  INJ1 25 41 1  41 OPEN 2* 0.152 3* Z/
  /

WCONPROD
  PROD1  OPEN  ORAT  800.00 4* 35.3/
  /


WCONINJE
  INJ1 WATER OPEN RATE 1200.00 1* 343.2 1* 0 /
  /
  
 TUNING
 0.01  15.0  0.1  0.2   2 /
                   /
  12 2 15          /

--WELOPEN
--'*'  SHUT /
--/
TSTEP
243*15
/
TSTEP
1*5
/

