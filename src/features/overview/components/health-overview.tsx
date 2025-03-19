// src/features/overview/components/health-overview.tsx
'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { healthAnalyticsData, recentSessions } from "@/constants/data";
import { Activity, Microscope, FileText, Pill, TrendingUp, Brain, Calendar, Users } from "lucide-react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, Tooltip, XAxis, YAxis, ResponsiveContainer } from "recharts";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { format } from "date-fns";
import Link from "next/link";

const chartConfig = {
  clinical: {
    label: 'Clinical Case Analysis',
    color: 'hsl(var(--chart-1))'
  },
  literature: {
    label: 'Medical Literature Review',
    color: 'hsl(var(--chart-2))'
  },
  symptom: {
    label: 'Symptom Analysis',
    color: 'hsl(var(--chart-3))'
  },
  drug: {
    label: 'Drug Interaction',
    color: 'hsl(var(--chart-4))'
  }
} satisfies ChartConfig;

// Data for the pie chart
const sessionTypeData = [
  { name: 'Clinical', value: 42, fill: 'hsl(var(--chart-1))' },
  { name: 'Literature', value: 28, fill: 'hsl(var(--chart-2))' },
  { name: 'Symptom', value: 33, fill: 'hsl(var(--chart-3))' },
  { name: 'Drug', value: 27, fill: 'hsl(var(--chart-4))' }
];

// Additional data for upcoming appointments
const upcomingAppointments = [
  { id: 1, patientName: "Alex Johnson", date: "2025-03-21T10:30:00", specialty: "Cardiology", status: "confirmed" },
  { id: 2, patientName: "Maya Patel", date: "2025-03-21T14:15:00", specialty: "Neurology", status: "confirmed" },
  { id: 3, patientName: "Robert Chen", date: "2025-03-22T09:00:00", specialty: "Orthopedics", status: "pending" },
  { id: 4, patientName: "Sarah Williams", date: "2025-03-23T11:30:00", specialty: "Dermatology", status: "confirmed" }
];

// Top conditions data
const topConditions = [
  { name: "Hypertension", count: 43, percentage: 23 },
  { name: "Type 2 Diabetes", count: 37, percentage: 19 },
  { name: "Migraine", count: 28, percentage: 15 },
  { name: "Anxiety", count: 24, percentage: 13 },
  { name: "Osteoarthritis", count: 19, percentage: 10 }
];

export default function HealthOverview() {
  return (
    <div className="flex flex-1 flex-col space-y-4">
      <Tabs defaultValue="overview" className="w-full">
        <div className="flex items-center justify-between">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="patients">Patients</TabsTrigger>
          </TabsList>
          
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">
              <Calendar className="mr-2 h-4 w-4" />
              Mar 15 - Mar 21
            </Button>
            <Button variant="default" size="sm">
              <TrendingUp className="mr-2 h-4 w-4" />
              Generate Report
            </Button>
          </div>
        </div>
      
        <TabsContent value="overview" className="space-y-4 mt-4">
          {/* Summary Cards */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Total Sessions
                </CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">130</div>
                <p className="text-xs text-muted-foreground">
                  +18.2% from last month
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Clinical Cases
                </CardTitle>
                <Microscope className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">42</div>
                <p className="text-xs text-muted-foreground">
                  +12.5% from last month
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Patients</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">85</div>
                <p className="text-xs text-muted-foreground">
                  +6.8% from last month
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  Appointments
                </CardTitle>
                <Calendar className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">27</div>
                <p className="text-xs text-muted-foreground">
                  +21.4% since last week
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-7">
            {/* Sessions Analytics */}
            <Card className="col-span-4">
              <CardHeader>
                <CardTitle>Session Analytics</CardTitle>
                <CardDescription>
                  Analysis sessions over the last 6 months
                </CardDescription>
              </CardHeader>
              <CardContent className="px-2 sm:p-6">
                <ChartContainer
                  config={chartConfig}
                  className="aspect-auto h-[300px] w-full"
                >
                  <AreaChart
                    accessibilityLayer
                    data={healthAnalyticsData}
                    margin={{
                      left: 12,
                      right: 12
                    }}
                  >
                    <CartesianGrid vertical={false} />
                    <XAxis
                      dataKey="month"
                      tickLine={false}
                      axisLine={false}
                      tickMargin={8}
                    />
                    <ChartTooltip
                      cursor={false}
                      content={<ChartTooltipContent indicator="dot" />}
                    />
                    <Area
                      dataKey="clinical"
                      type="monotone"
                      fill="var(--color-clinical)"
                      fillOpacity={0.4}
                      stroke="var(--color-clinical)"
                      stackId="1"
                    />
                    <Area
                      dataKey="literature"
                      type="monotone"
                      fill="var(--color-literature)"
                      fillOpacity={0.4}
                      stroke="var(--color-literature)"
                      stackId="1"
                    />
                    <Area
                      dataKey="symptom"
                      type="monotone"
                      fill="var(--color-symptom)"
                      fillOpacity={0.4}
                      stroke="var(--color-symptom)"
                      stackId="1"
                    />
                    <Area
                      dataKey="drug"
                      type="monotone"
                      fill="var(--color-drug)"
                      fillOpacity={0.4}
                      stroke="var(--color-drug)"
                      stackId="1"
                    />
                  </AreaChart>
                </ChartContainer>
              </CardContent>
            </Card>

            {/* Recent Sessions Card */}
            <Card className="col-span-3">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Recent Sessions</CardTitle>
                  <CardDescription>
                    Your most recent analysis sessions
                  </CardDescription>
                </div>
                <Link href="/dashboard/chat">
                  <Button variant="outline" size="sm">View All</Button>
                </Link>
              </CardHeader>
              <CardContent>
                <div className="space-y-8">
                  {recentSessions.map((session) => (
                    <div key={session.id} className="flex items-center">
                      <Avatar className="h-9 w-9">
                        <AvatarFallback className={cn(
                          "text-primary-foreground",
                          session.type.includes("Clinical") && "bg-[hsl(var(--chart-1))]",
                          session.type.includes("Literature") && "bg-[hsl(var(--chart-2))]",
                          session.type.includes("Symptom") && "bg-[hsl(var(--chart-3))]",
                          session.type.includes("Drug") && "bg-[hsl(var(--chart-4))]"
                        )}>
                          {session.type.substring(0, 2)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="ml-4 space-y-1">
                        <p className="text-sm font-medium leading-none">{session.type}</p>
                        <p className="text-sm text-muted-foreground">
                          {session.summary}
                        </p>
                      </div>
                      <div className="ml-auto">
                        <Badge
                          variant={
                            session.status === 'completed' 
                              ? 'outline' 
                              : session.status === 'in-progress' 
                                ? 'secondary' 
                                : 'default'
                          }
                          className={cn(
                            "ml-2",
                            session.status === 'completed' && "border-green-500 text-green-500",
                            session.status === 'in-progress' && "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
                            session.status === 'scheduled' && "bg-primary"
                          )}
                        >
                          {session.status === 'completed' && "Completed"}
                          {session.status === 'in-progress' && "In Progress"}
                          {session.status === 'scheduled' && "Scheduled"}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Monthly Breakdown */}
            <Card className="col-span-4">
              <CardHeader>
                <CardTitle>Monthly Breakdown</CardTitle>
                <CardDescription>
                  Session counts by type per month
                </CardDescription>
              </CardHeader>
              <CardContent className="px-2 sm:p-6">
                <ChartContainer
                  config={chartConfig}
                  className="aspect-auto h-[300px] w-full"
                >
                  <BarChart
                    accessibilityLayer
                    data={healthAnalyticsData}
                    margin={{
                      left: 12,
                      right: 12
                    }}
                  >
                    <CartesianGrid vertical={false} />
                    <XAxis
                      dataKey="month"
                      tickLine={false}
                      axisLine={false}
                      tickMargin={8}
                    />
                    <ChartTooltip
                      content={<ChartTooltipContent className="w-[150px]" />}
                    />
                    <Bar dataKey="clinical" fill="var(--color-clinical)" />
                    <Bar dataKey="literature" fill="var(--color-literature)" />
                    <Bar dataKey="symptom" fill="var(--color-symptom)" />
                    <Bar dataKey="drug" fill="var(--color-drug)" />
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>

            {/* Distribution Pie Chart */}
            <Card className="col-span-3">
              <CardHeader>
                <CardTitle>Session Distribution</CardTitle>
                <CardDescription>
                  Breakdown by session type
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-1 pb-0">
                <ChartContainer
                  config={chartConfig}
                  className="mx-auto aspect-square max-h-[360px]"
                >
                  <PieChart>
                    <ChartTooltip content={<ChartTooltipContent hideLabel />} />
                    <Pie
                      data={sessionTypeData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={90}
                      fill="#8884d8"
                    >
                      {sessionTypeData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Legend />
                  </PieChart>
                </ChartContainer>
              </CardContent>
            </Card>
            
            {/* Additional Cards for Professional Dashboard */}
            <Card className="col-span-4">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Upcoming Appointments</CardTitle>
                  <CardDescription>
                    Scheduled for the next 3 days
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm">
                  <Calendar className="mr-2 h-4 w-4" />
                  Schedule New
                </Button>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {upcomingAppointments.map((appointment) => (
                    <div key={appointment.id} className="flex items-center justify-between border-b pb-4 last:border-0 last:pb-0">
                      <div className="flex items-center space-x-4">
                        <Avatar className="h-9 w-9">
                          <AvatarFallback className="bg-primary text-primary-foreground">
                            {appointment.patientName.split(' ').map(n => n[0]).join('')}
                          </AvatarFallback>
                        </Avatar>
                        <div>
                          <p className="text-sm font-medium">{appointment.patientName}</p>
                          <p className="text-xs text-muted-foreground">{appointment.specialty}</p>
                        </div>
                      </div>
                      <div className="text-sm">
                        {format(new Date(appointment.date), "MMM d, h:mm a")}
                      </div>
                      <Badge variant={appointment.status === 'confirmed' ? 'default' : 'outline'}>
                        {appointment.status === 'confirmed' ? 'Confirmed' : 'Pending'}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            {/* Top Conditions Card */}
            <Card className="col-span-3">
              <CardHeader>
                <CardTitle>Top Conditions</CardTitle>
                <CardDescription>
                  Most frequent patient conditions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {topConditions.map((condition) => (
                    <div key={condition.name} className="flex items-center">
                      <div className="flex-1">
                        <div className="flex items-center">
                          <span className="text-sm font-medium">{condition.name}</span>
                          <span className="ml-auto text-sm text-muted-foreground">{condition.count} patients</span>
                        </div>
                        <div className="mt-1 h-2 w-full rounded-full bg-secondary">
                          <div 
                            className="h-full rounded-full bg-primary" 
                            style={{ width: `${condition.percentage}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-4 mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Advanced Analytics</CardTitle>
              <CardDescription>
                Detailed metrics and trends for your healthcare practice
              </CardDescription>
            </CardHeader>
            <CardContent className="h-[400px] flex items-center justify-center text-muted-foreground">
              Analytics dashboard would be displayed here
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Patients Tab */}
        <TabsContent value="patients" className="space-y-4 mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Patient Management</CardTitle>
              <CardDescription>
                View and manage your patient records
              </CardDescription>
            </CardHeader>
            <CardContent className="h-[400px] flex items-center justify-center text-muted-foreground">
              Patient management dashboard would be displayed here
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}