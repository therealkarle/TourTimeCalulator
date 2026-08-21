# Beispielbefehle für Tour-Schätzungen

Die folgenden Befehle können in PowerShell im Projektverzeichnis ausgeführt werden.

## LongRunMin15kmAllTime (Ridge)

```powershell
.\run.bat estimate --model "longrunmin15kmalltime_ridge" --distance-km 21.4 --elevation-up-m 220 --elevation-down-m 220
```

## TrailRunMin8km (Linear)

```powershell
.\run.bat estimate --model "trailrunmin8km_linear" --distance-km 17.6 --elevation-up-m 1590 --elevation-down-m 1140
```

## TrailRunMin8km (Ridge)

```powershell
.\run.bat estimate --model "trailrunmin8km_ridge" --distance-km 17.6 --elevation-up-m 1590 --elevation-down-m 1140
```

## ride_withpowerData_all (Ridge)

```powershell
.\run.bat estimate --model "ride_withpowerdata_all_ridge" --distance-km 135 --elevation-up-m 2280 --elevation-down-m 2280
```
