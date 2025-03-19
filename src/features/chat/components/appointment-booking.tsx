'use client';

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Calendar } from 'lucide-react';
import { format } from 'date-fns';

interface Specialist {
  id: number;
  name: string;
  specialty: string;
  hospital: string;
  experience: number;
  bio: string;
  image_url: string;
  available_slots: {
    doctor_id: number;
    start_time: string;
    end_time: string;
    formatted_time: string;
    is_booked: boolean;
  }[];
}

enum BookingStep {
  SelectDoctor = 0,
  SelectTime = 1,
  EnterDetails = 2,
  Confirmation = 3,
  Success = 4
}

interface AppointmentBookingProps {
  specialists: Specialist[];
}

export default function AppointmentBooking({ specialists }: AppointmentBookingProps) {
  const [currentStep, setCurrentStep] = useState(BookingStep.SelectDoctor);
  const [selectedDoctor, setSelectedDoctor] = useState<Specialist | null>(null);
  const [selectedTimeSlot, setSelectedTimeSlot] = useState<any | null>(null);
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [notes, setNotes] = useState('');
  const [isBooking, setIsBooking] = useState(false);
  const [bookingComplete, setBookingComplete] = useState(false);
  const [meetLink, setMeetLink] = useState('');
  
  const handleSelectDoctor = (doctor: Specialist) => {
    setSelectedDoctor(doctor);
    setCurrentStep(BookingStep.SelectTime);
  };
  
  const handleSelectTime = (timeSlot: any) => {
    setSelectedTimeSlot(timeSlot);
    setCurrentStep(BookingStep.EnterDetails);
  };
  
  const handleSubmitDetails = () => {
    if (!email || !name) return;
    setCurrentStep(BookingStep.Confirmation);
  };
  
  const handleConfirmBooking = async () => {
    if (!selectedDoctor || !selectedTimeSlot) return;
    
    setIsBooking(true);
    
    try {
      const response = await fetch('/api/book-appointment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          doctor_id: selectedDoctor.id,
          doctor_name: selectedDoctor.name,
          slot_id: selectedTimeSlot.formatted_time,
          email,
          name,
          phone,
          notes,
          specialists
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to book appointment');
      }
      
      const data = await response.json();
      setMeetLink(data.appointment.google_meet_link);
      setBookingComplete(true);
      setCurrentStep(BookingStep.Success);
    } catch (error) {
      console.error('Error booking appointment:', error);
      alert('Failed to book appointment. Please try again.');
    } finally {
      setIsBooking(false);
    }
  };
  
  const resetBooking = () => {
    setCurrentStep(BookingStep.SelectDoctor);
    setSelectedDoctor(null);
    setSelectedTimeSlot(null);
    setEmail('');
    setName('');
    setPhone('');
    setNotes('');
    setBookingComplete(false);
  };
  
  const goBack = () => {
    setCurrentStep(prev => prev > 0 ? prev - 1 : prev);
  };
  
  return (
    <div className="booking-container mt-2 border rounded-lg p-4">
      <div className="mb-4">
        <h3 className="text-lg font-medium flex items-center">
          <Calendar className="mr-2 h-5 w-5 text-primary" />
          Book an Appointment with a Specialist
        </h3>
      </div>
      
      {/* Steps indicator */}
      <div className="mb-6">
        <div className="flex justify-between">
          {['Select Doctor', 'Choose Time', 'Your Details', 'Confirm'].map((step, index) => (
            <div 
              key={step}
              className={`flex flex-col items-center ${index <= currentStep ? 'text-primary' : 'text-gray-400'}`}
            >
              <div 
                className={`w-8 h-8 rounded-full flex items-center justify-center mb-1 
                ${index < currentStep ? 'bg-primary text-white' : 
                  index === currentStep ? 'border-2 border-primary text-primary' : 
                  'border-2 border-gray-300 text-gray-400'}`}
              >
                {index + 1}
              </div>
              <span className="text-xs hidden sm:inline">{step}</span>
            </div>
          ))}
        </div>
        <div className="w-full h-1 bg-gray-200 mt-4 mb-8 relative">
          <div 
            className="absolute h-1 bg-primary transition-all duration-300" 
            style={{ width: `${(currentStep / 3) * 100}%` }}
          ></div>
        </div>
      </div>
      
      {/* Step content */}
      <div className="mt-4">
        {/* Step 1: Select Doctor */}
        {currentStep === BookingStep.SelectDoctor && (
          <div className="space-y-4">
            <h4 className="text-md font-medium">Select a Doctor</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {specialists.map((doctor) => (
                <div 
                  key={doctor.id} 
                  className={`border rounded-lg p-4 cursor-pointer hover:border-primary transition-colors ${selectedDoctor?.id === doctor.id ? 'border-primary bg-primary/5' : ''}`}
                  onClick={() => handleSelectDoctor(doctor)}
                >
                  <div className="flex items-start space-x-3">
                    <Avatar className="h-12 w-12">
                      <AvatarImage src={doctor.image_url} alt={doctor.name} />
                      <AvatarFallback>{doctor.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                    </Avatar>
                    <div>
                      <h5 className="font-medium">{doctor.name}</h5>
                      <p className="text-sm text-primary">{doctor.specialty}</p>
                      <p className="text-xs text-gray-500">{doctor.hospital}</p>
                      <p className="text-xs">{doctor.experience} years experience</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Step 2: Select Time */}
        {currentStep === BookingStep.SelectTime && selectedDoctor && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h4 className="text-md font-medium">Select Appointment Time</h4>
              <Button variant="ghost" size="sm" onClick={goBack}>
                Back
              </Button>
            </div>
            
            <div className="border rounded-lg p-4 mb-4">
              <div className="flex items-start space-x-3">
                <Avatar className="h-10 w-10">
                  <AvatarImage src={selectedDoctor.image_url} alt={selectedDoctor.name} />
                  <AvatarFallback>{selectedDoctor.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                </Avatar>
                <div>
                  <h5 className="font-medium">{selectedDoctor.name}</h5>
                  <p className="text-sm text-primary">{selectedDoctor.specialty}</p>
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
              {selectedDoctor.available_slots.map((slot) => (
                <div 
                  key={slot.formatted_time}
                  className={`border rounded p-2 text-center cursor-pointer ${
                    slot.is_booked ? 'bg-gray-100 text-gray-400 cursor-not-allowed' : 
                    selectedTimeSlot?.formatted_time === slot.formatted_time ? 'border-primary bg-primary/5' : 
                    'hover:border-primary'
                  }`}
                  onClick={() => !slot.is_booked && handleSelectTime(slot)}
                >
                  <p className="text-sm">{slot.formatted_time}</p>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Step 3: Enter Details */}
        {currentStep === BookingStep.EnterDetails && selectedDoctor && selectedTimeSlot && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h4 className="text-md font-medium">Your Contact Information</h4>
              <Button variant="ghost" size="sm" onClick={goBack}>
                Back
              </Button>
            </div>
            
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium mb-1">Full Name *</label>
                <Input 
                  value={name} 
                  onChange={(e) => setName(e.target.value)} 
                  placeholder="Enter your full name"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Email Address *</label>
                <Input 
                  type="email" 
                  value={email} 
                  onChange={(e) => setEmail(e.target.value)} 
                  placeholder="your.email@example.com"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Phone Number</label>
                <Input 
                  type="tel" 
                  value={phone} 
                  onChange={(e) => setPhone(e.target.value)} 
                  placeholder="(123) 456-7890"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Notes (Optional)</label>
                <Input 
                  value={notes} 
                  onChange={(e) => setNotes(e.target.value)} 
                  placeholder="Any additional notes for the doctor"
                />
              </div>
            </div>
            
            <Button 
              className="w-full mt-4" 
              onClick={handleSubmitDetails}
              disabled={!email || !name}
            >
              Continue to Confirmation
            </Button>
          </div>
        )}
        
        {/* Step 4: Confirmation */}
        {currentStep === BookingStep.Confirmation && selectedDoctor && selectedTimeSlot && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h4 className="text-md font-medium">Confirm Your Appointment</h4>
              <Button variant="ghost" size="sm" onClick={goBack}>
                Back
              </Button>
            </div>
            
            <div className="border rounded-lg p-4 space-y-3">
              <div className="flex justify-between">
                <span className="text-sm font-medium">Doctor:</span>
                <span className="text-sm">{selectedDoctor.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Specialty:</span>
                <span className="text-sm">{selectedDoctor.specialty}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Date & Time:</span>
                <span className="text-sm">{selectedTimeSlot.formatted_time}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Patient:</span>
                <span className="text-sm">{name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Email:</span>
                <span className="text-sm">{email}</span>
              </div>
              {phone && (
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Phone:</span>
                  <span className="text-sm">{phone}</span>
                </div>
              )}
            </div>
            
            <Button 
              className="w-full mt-4" 
              onClick={handleConfirmBooking}
              disabled={isBooking}
            >
              {isBooking ? 'Booking...' : 'Confirm Appointment'}
            </Button>
          </div>
        )}
        
        {/* Step 5: Success */}
        {currentStep === BookingStep.Success && (
          <div className="space-y-4 text-center">
            <div className="flex justify-center items-center h-16 w-16 mx-auto bg-green-100 rounded-full">
              <svg className="h-8 w-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            
            <h4 className="text-lg font-medium">Appointment Confirmed!</h4>
            <p className="text-sm text-gray-600">
              Your appointment with {selectedDoctor?.name} on {selectedTimeSlot?.formatted_time} has been confirmed.
            </p>
            
            <div className="border rounded-lg p-4 text-left space-y-2">
              <div className="flex justify-between">
                <span className="text-sm font-medium">Google Meet Link:</span>
                <a href={meetLink} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline">
                  Join Meeting
                </a>
              </div>
            </div>
            
            <p className="text-sm text-gray-600 mt-4">
              A confirmation email has been sent to {email} with all the details.
            </p>
            
            <Button 
              className="w-full mt-4" 
              variant="outline"
              onClick={resetBooking}
            >
              Book Another Appointment
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}